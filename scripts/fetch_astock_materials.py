#!/usr/bin/env python3
"""Fetch a conservative material bundle using a-stock-data's public endpoints.

This is a small, auditable subset of simonlin1212/a-stock-data v3.5.0:
company news, CNINFO announcements/IRM, company/industry research metadata,
CLS telegraphs and Eastmoney global news. Eastmoney calls are serialized and
rate-limited. Documents without an exact intraday publish time are assigned
23:59:59, so a same-day close decision cannot accidentally see them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import requests
    from bs4 import BeautifulSoup
    from pypdf import PdfReader
except ImportError as exc:
    raise SystemExit(
        "缺少 requests；请先运行: python -m pip install -r requirements-materials.txt"
    ) from exc

TZ = ZoneInfo("Asia/Shanghai")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
REPORT_API = "https://reportapi.eastmoney.com/report/list"
_last_em_call = 0.0
_cninfo_orgids: dict[str, str] = {}


def _em_get(url: str, **kwargs) -> requests.Response:
    global _last_em_call
    delay = 1.1 + random.random() * 0.4 - (time.monotonic() - _last_em_call)
    if delay > 0:
        time.sleep(delay)
    response = requests.get(url, **kwargs)
    _last_em_call = time.monotonic()
    response.raise_for_status()
    return response


def _source_id(kind: str, *parts: Any) -> str:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:20]
    return f"{kind}-{digest}"


def _published(value: Any, exact_time: bool = True) -> str:
    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(value / (1000 if value > 10**11 else 1), TZ)
    else:
        text = str(value or "").strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ)
    if not exact_time and dt.hour == dt.minute == dt.second == 0:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.isoformat()


def _doc(source_id: str, published_at: str, source_type: str,
         title: str, content: str, url: str = "", **metadata) -> dict[str, Any]:
    return {
        "source_id": source_id, "published_at": published_at,
        "source_type": source_type, "title": title,
        "content": content, "url": url, "metadata": metadata,
    }


def company_news(code: str, page_size: int) -> list[dict[str, Any]]:
    cb = "jQuery_news"
    inner = json.dumps({
        "uid": "", "keyword": code, "type": ["cmsArticleWebOld"],
        "client": "web", "clientType": "web", "clientVersion": "curr",
        "param": {"cmsArticleWebOld": {
            "searchScope": "default", "sort": "default", "pageIndex": 1,
            "pageSize": page_size, "preTag": "", "postTag": ""}},
    }, separators=(",", ":"))
    response = _em_get(
        "https://search-api-web.eastmoney.com/search/jsonp",
        params={"cb": cb, "param": inner},
        headers={"User-Agent": UA, "Referer": "https://so.eastmoney.com/"},
        timeout=20)
    text = response.text
    payload = json.loads(text[text.index("(") + 1:text.rindex(")")])
    result = []
    for row in payload.get("result", {}).get("cmsArticleWebOld", []) or []:
        title = re.sub(r"<[^>]+>", "", row.get("title", ""))
        content = re.sub(r"<[^>]+>", "", row.get("content", ""))
        url = row.get("url", "")
        result.append(_doc(
            _source_id("em-news", url, title), _published(row.get("date")),
            "financial_news", title, content, url, media=row.get("mediaName", "")))
    return result


def _cninfo_orgid(code: str) -> str:
    global _cninfo_orgids
    if not _cninfo_orgids:
        response = requests.get(
            "http://www.cninfo.com.cn/new/data/szse_stock.json",
            headers={"User-Agent": UA}, timeout=20)
        response.raise_for_status()
        _cninfo_orgids = {
            row["code"]: row["orgId"] for row in response.json().get("stockList", [])}
    return _cninfo_orgids.get(
        code, f"gssh0{code}" if code.startswith("6") else f"gssz0{code}")


def announcements(code: str, page_size: int) -> list[dict[str, Any]]:
    response = requests.post(
        "https://www.cninfo.com.cn/new/hisAnnouncement/query",
        data={
            "stock": f"{code},{_cninfo_orgid(code)}", "tabName": "fulltext",
            "pageSize": str(page_size), "pageNum": "1", "column": "",
            "category": "", "plate": "", "seDate": "", "searchkey": "",
            "secid": "", "sortName": "", "sortType": "", "isHLtitle": "true",
        },
        headers={
            "User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded",
            "Referer": "https://www.cninfo.com.cn/new/disclosure",
            "Origin": "https://www.cninfo.com.cn",
        }, timeout=20)
    response.raise_for_status()
    result = []
    for row in response.json().get("announcements", []) or []:
        announcement_id = row.get("announcementId", "")
        title = row.get("announcementTitle", "")
        url = f"https://www.cninfo.com.cn/new/disclosure/detail?annoId={announcement_id}"
        adjunct = row.get("adjunctUrl", "")
        pdf_url = (
            f"https://static.cninfo.com.cn/{adjunct.lstrip('/')}" if adjunct else "")
        # CNINFO exposes millisecond time. Preserve it if intraday; otherwise
        # conservative end-of-day handling is applied by _published.
        raw_time = row.get("announcementTime")
        published = _published(raw_time, exact_time=False)
        result.append(_doc(
            _source_id("cninfo", announcement_id, title), published,
            "exchange_announcement", title,
            f"公告类型：{row.get('announcementTypeName', '')}；标题：{title}",
            url, announcement_id=announcement_id, pdf_url=pdf_url))
    return result


def irm(code: str, page_size: int) -> list[dict[str, Any]]:
    first = requests.post(
        "https://irm.cninfo.com.cn/newircs/index/queryKeyboardInfo",
        data={"keyWord": code}, headers={"User-Agent": UA}, timeout=15)
    first.raise_for_status()
    matches = first.json().get("data") or []
    if not matches:
        return []
    second = requests.post(
        "https://irm.cninfo.com.cn/newircs/company/question",
        params={
            "_t": 1, "stockcode": code, "orgId": matches[0].get("secid"),
            "pageSize": page_size, "pageNum": 1, "keyWord": "",
            "startDay": "", "endDay": "",
        }, headers={"User-Agent": UA}, timeout=15)
    second.raise_for_status()
    result = []
    for row in second.json().get("rows") or []:
        if not row.get("attachedContent"):
            continue
        question, answer = row.get("mainContent", ""), row.get("attachedContent", "")
        result.append(_doc(
            _source_id("irm", code, row.get("pubDate"), question),
            _published(row.get("pubDate")), "company_ir",
            f"互动问答：{question[:80]}", f"投资者问：{question}\n公司答：{answer}",
            answerer=row.get("attachedAuthor", "")))
    return result


def reports(code: str, page_size: int, industry: bool = False) -> list[dict[str, Any]]:
    params = {
        "industryCode": "*", "pageSize": str(page_size), "industry": "*",
        "rating": "*", "ratingChange": "*", "beginTime": "2020-01-01",
        "endTime": "2030-01-01", "pageNo": "1", "fields": "",
        "qType": "1" if industry else "0",
    }
    if not industry:
        params["code"] = code
    payload = _em_get(
        REPORT_API, params=params,
        headers={"User-Agent": UA, "Referer": "https://data.eastmoney.com/"},
        timeout=30).json()
    result = []
    for row in payload.get("data") or []:
        info = row.get("infoCode", "")
        title = row.get("title", "")
        content = (
            f"机构：{row.get('orgSName', '')}；评级：{row.get('emRatingName', '')}；"
            f"行业：{row.get('industryName') or row.get('indvInduName', '')}；"
            f"本年EPS预测：{row.get('predictThisYearEps')}；"
            f"明年EPS预测：{row.get('predictNextYearEps')}；标题：{title}"
        )
        result.append(_doc(
            _source_id("industry-report" if industry else "stock-report", info, title),
            _published(row.get("publishDate"), exact_time=False),
            "research_report", title, content,
            f"https://pdf.dfcfw.com/pdf/H3_{info}_1.pdf" if info else "",
            institution=row.get("orgSName", ""),
            industry=row.get("industryName") or row.get("indvInduName", "")))
    return result


def cls_macro(page_size: int) -> list[dict[str, Any]]:
    params = {
        "appName": "CailianpressWeb", "os": "web", "sv": "7.7.5",
        "last_time": "", "refresh_type": "1", "rn": str(page_size)}
    query = "&".join(f"{key}={params[key]}" for key in sorted(params))
    sign = hashlib.md5(hashlib.sha1(query.encode()).hexdigest().encode()).hexdigest()
    response = requests.get(
        f"https://www.cls.cn/v1/roll/get_roll_list?{query}&sign={sign}",
        headers={"User-Agent": UA, "Referer": "https://www.cls.cn/"}, timeout=15)
    response.raise_for_status()
    result = []
    for row in response.json().get("data", {}).get("roll_data", []) or []:
        title = row.get("title", "") or row.get("brief", "")
        content = row.get("content", "") or row.get("brief", "")
        result.append(_doc(
            _source_id("cls", row.get("id"), row.get("ctime"), title),
            _published(row.get("ctime")), "financial_news", title, content,
            scope="macro_market"))
    return result


def global_macro(page_size: int) -> list[dict[str, Any]]:
    response = _em_get(
        "https://np-weblist.eastmoney.com/comm/web/getFastNewsList",
        params={
            "client": "web", "biz": "web_724", "fastColumn": "102",
            "sortEnd": "", "pageSize": str(page_size),
            "req_trace": str(uuid.uuid4()),
        }, headers={"User-Agent": UA, "Referer": "https://kuaixun.eastmoney.com/"},
        timeout=15)
    result = []
    for row in response.json().get("data", {}).get("fastNewsList", []) or []:
        title, summary = row.get("title", ""), row.get("summary", "")
        result.append(_doc(
            _source_id("em-global", row.get("code"), row.get("showTime"), title),
            _published(row.get("showTime")), "financial_news", title, summary,
            scope="macro_market"))
    return result


def _extract_pdf(content: bytes, target: Path) -> tuple[str, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    reader = PdfReader(target)
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return text.strip(), hashlib.sha256(content).hexdigest()


def hydrate_full_text(
    document: dict[str, Any], raw_dir: Path, timeout: int = 60
) -> dict[str, Any]:
    """Download and parse an announcement/report PDF or a news HTML page."""
    result = dict(document)
    metadata = dict(result.get("metadata") or {})
    source_type = result["source_type"]
    pdf_url = metadata.get("pdf_url")
    if source_type == "research_report" and result.get("url", "").endswith(".pdf"):
        pdf_url = result["url"]
    try:
        if pdf_url:
            response = requests.get(
                pdf_url, headers={"User-Agent": UA, "Referer": result.get("url", "")},
                timeout=timeout)
            response.raise_for_status()
            target = raw_dir / f"{result['source_id']}.pdf"
            text, digest = _extract_pdf(response.content, target)
            if text:
                result["content"] = text
            metadata.update({
                "raw_path": str(target), "content_sha256": digest,
                "parser": "pypdf", "text_length": len(text),
            })
        elif source_type == "financial_news" and result.get("url"):
            response = requests.get(
                result["url"], headers={"User-Agent": UA}, timeout=timeout)
            response.raise_for_status()
            target = raw_dir / f"{result['source_id']}.html"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.content)
            soup = BeautifulSoup(response.content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = "\n".join(
                line.strip() for line in soup.get_text("\n").splitlines()
                if line.strip())
            if len(text) > len(result.get("content", "")):
                result["content"] = text
            metadata.update({
                "raw_path": str(target),
                "content_sha256": hashlib.sha256(response.content).hexdigest(),
                "parser": "beautifulsoup4", "text_length": len(text),
            })
    except Exception as exc:
        metadata["full_text_error"] = f"{type(exc).__name__}: {exc}"
    result["metadata"] = metadata
    return result


def _clean_surrogates(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, list):
        return [_clean_surrogates(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_surrogates(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="6位代码或带交易所后缀")
    parser.add_argument("--decision-time", required=True, help="带时区 ISO 时间")
    parser.add_argument("--start-time", help="可选材料窗口起点，带时区 ISO 时间")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--skip-industry-reports", action="store_true")
    parser.add_argument("--full-text-dir", type=Path)
    args = parser.parse_args()
    code = args.ticker.split(".")[0]
    decision = datetime.fromisoformat(args.decision_time)
    if decision.tzinfo is None:
        raise ValueError("--decision-time 必须包含时区")
    start = datetime.fromisoformat(args.start_time) if args.start_time else None
    if start is not None and start.tzinfo is None:
        raise ValueError("--start-time 必须包含时区")

    fetched: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    fetchers = {
        "company_news": lambda: company_news(code, args.page_size),
        "announcements": lambda: announcements(code, args.page_size),
        "irm": lambda: irm(code, args.page_size),
        "company_reports": lambda: reports(code, args.page_size),
        "cls_macro": lambda: cls_macro(args.page_size),
        "global_macro": lambda: global_macro(args.page_size),
    }
    if not args.skip_industry_reports:
        fetchers["industry_reports"] = lambda: reports(code, args.page_size, industry=True)
    for name, fetch in fetchers.items():
        try:
            fetched.extend(fetch())
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"

    accepted = [
        row for row in fetched
        if datetime.fromisoformat(row["published_at"]) <= decision
        and (start is None or datetime.fromisoformat(row["published_at"]) >= start)
    ]
    unique = {row["source_id"]: row for row in accepted}
    if args.full_text_dir:
        hydrated = {}
        for source_id, row in unique.items():
            hydrated[source_id] = hydrate_full_text(
                row, args.full_text_dir / code)
        unique = hydrated
    payload = {
        "ticker": args.ticker, "decision_time": decision.isoformat(),
        "documents": list(unique.values()),
        "fetch_metadata": {
            "fetched_at": datetime.now(TZ).isoformat(),
            "fetched_count": len(fetched), "accepted_count": len(unique),
            "future_rejected_count": len(fetched) - len(accepted),
            "errors": errors,
            "upstream_reference": "simonlin1212/a-stock-data v3.5.0",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = _clean_surrogates(payload)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(payload["fetch_metadata"], ensure_ascii=False))


if __name__ == "__main__":
    main()
