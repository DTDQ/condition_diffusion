#!/usr/bin/env python3
"""Extract point-in-time stock events with the GLM HTTP API.

The API key is read only from GLM_API_KEY. GLM's JSON mode guarantees valid
JSON; the repository's schema and evidence/time checks are enforced locally.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import requests
    import jsonschema
except ImportError as exc:
    raise SystemExit(
        "缺少依赖；请先运行: python -m pip install -r requirements-materials.txt"
    ) from exc

sys.path.insert(0, str(Path(__file__).parent))
from four_layer_conditions import Event, filter_point_in_time  # noqa: E402

GLM_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-4.7-flashx"
EVENT_KEYS = {
    "event_type", "layer", "direction", "magnitude", "exposure",
    "horizon_days", "novelty", "source_quality", "evidence_confidence",
    "published_at", "source_type", "source_ids", "evidence_text",
    "actual", "expected",
}
REQUIRED_SEMANTIC = {
    "event_type", "layer", "direction", "magnitude", "exposure",
    "horizon_days", "novelty", "source_quality", "evidence_confidence",
}

SYSTEM_PROMPT = """你是金融材料事件抽取器，不是股票预测员。
你只能依据用户提供的材料工作。禁止使用模型记忆补充事实，禁止推测后续股价。
你的全部回复必须是符合用户给定 JSON Schema 的一个 JSON 对象，不要使用 Markdown。"""

USER_PROMPT = """任务：只根据 <materials> 中的材料，为 {ticker} 在决策时点
{decision_time} 生成事件 JSON。

强制规则：
1. published_at 晚于决策时点的材料不得输出。
2. 每个事件必须引用材料中真实存在的 source_id，并给出简短证据原文。
3. 不得补全缺失数字；没有 actual/expected 时填 null。
4. direction 表示事件对目标公司未来基本面或风险的方向，不是股价预测。
5. magnitude 是事件本身强度，exposure 是目标公司暴露度；不确定时降低
   evidence_confidence。
6. 相同事实的转载合并为一个事件，source_ids 可以包含多个来源。
7. 研报观点不能当成公司已发生事实，source_type 必须为 research_report。
8. 宏观事件必须有材料支持的公司或行业传导关系。
9. 没有可靠事件时返回 {{"events":[]}}。
10. 最多输出 24 个最重要且不重复的事件；证据原文不超过 120 个字符。
11. 只输出 Schema 要求的字段，不要解释、总结、推理过程或其他内容。

必须严格符合以下 JSON Schema：
<schema>
{schema}
</schema>

<materials>
{materials}
</materials>
"""

FINANCIAL_KEYWORDS = (
    "营业收入", "营收", "净利润", "扣非", "毛利率", "净利率", "现金流",
    "业绩", "预告", "指引", "同比", "环比", "订单", "合同", "中标",
    "产品", "技术", "研发", "产能", "产量", "销量", "价格", "库存",
    "供应链", "原材料", "管理层", "董事", "高管", "诉讼", "仲裁",
    "监管", "处罚", "立案", "并购", "重组", "回购", "增发", "减持",
    "解禁", "分红", "政策", "补贴", "风险", "不确定性", "预计", "预测",
)


def compact_financial_text(
    text: str, title: str, max_chars: int = 12_000
) -> str:
    """Keep auditable event-bearing passages while retaining full text on disk."""
    if len(text) <= max_chars:
        return text
    paragraphs = [
        re.sub(r"\s+", " ", part).strip()
        for part in re.split(r"[\r\n]+", text)
        if len(part.strip()) >= 12
    ]
    title_terms = {
        term for term in re.findall(r"[\u4e00-\u9fff]{2,6}", title)
        if len(term) >= 2
    }
    scored: list[tuple[float, int, str]] = []
    for index, part in enumerate(paragraphs):
        keyword_hits = sum(term in part for term in FINANCIAL_KEYWORDS)
        number_hits = min(len(re.findall(r"\d+(?:\.\d+)?%?", part)), 8)
        title_hits = min(sum(term in part for term in title_terms), 4)
        heading_bonus = 2 if re.match(r"^(第[一二三四五六七八九十]+|[一二三四五六七八九十]+、|\d+[.、])", part) else 0
        early_bonus = 2 if index < 12 else 0
        score = keyword_hits * 4 + number_hits + title_hits * 2 + heading_bonus + early_bonus
        if score > 0:
            scored.append((score, index, part))
    selected: dict[int, str] = {}
    used = 0
    for _, index, part in sorted(scored, key=lambda item: (-item[0], item[1])):
        if used + len(part) > max_chars:
            continue
        selected[index] = part
        used += len(part) + 1
        if used >= max_chars * 0.95:
            break
    if not selected:
        return text[:max_chars]
    return "\n".join(selected[index] for index in sorted(selected))


def compact_documents(
    documents: list[dict[str, Any]], max_chars_per_document: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    compacted = []
    before = after = 0
    for document in documents:
        row = dict(document)
        content = str(row.get("content", ""))
        before += len(content)
        if row.get("source_type") in {
            "exchange_announcement", "research_report", "financial_news",
        }:
            content = compact_financial_text(
                content, str(row.get("title", "")), max_chars_per_document)
        else:
            content = content[:max_chars_per_document]
        row["content"] = content
        after += len(content)
        compacted.append(row)
    return compacted, {"input_chars_before": before, "input_chars_after": after}


def load_local_env(path: Path) -> None:
    """Load simple KEY=VALUE lines without overriding process environment."""
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _parse_decision(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("--decision-time 必须包含时区")
    return result


def _validate_material_bundle(
    payload: Any, decision: datetime
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("documents"), list):
        raise ValueError('材料文件必须是 {"documents": [...]} 格式')
    accepted = []
    for doc in payload["documents"]:
        required = {"source_id", "published_at", "source_type", "title", "content"}
        missing = required - set(doc)
        if missing:
            raise ValueError(f"材料缺少字段: {sorted(missing)}")
        published = datetime.fromisoformat(doc["published_at"])
        if published.tzinfo is None:
            raise ValueError(f"材料 {doc['source_id']} 的 published_at 缺少时区")
        if published <= decision:
            accepted.append(doc)
    return accepted


def _extract_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("GLM 返回的顶层结构不是 JSON object")
    return payload


def call_glm(
    prompt: str,
    api_key: str,
    model: str,
    timeout: int = 180,
    max_attempts: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "do_sample": False,
        "thinking": {"type": "disabled"},
        "max_tokens": 2048,
        "stream": False,
        "request_id": str(uuid.uuid4()),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                GLM_ENDPOINT, headers=headers, json=request_body, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"GLM transient HTTP {response.status_code}: {response.text[:500]}")
            response.raise_for_status()
            envelope = response.json()
            content = envelope["choices"][0]["message"]["content"]
            return _extract_json(content), {
                "request_id": envelope.get("request_id") or envelope.get("id"),
                "model": envelope.get("model", model),
                "usage": envelope.get("usage", {}),
            }
        except (requests.RequestException, RuntimeError, KeyError, ValueError,
                json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"GLM 调用在 {max_attempts} 次尝试后失败: {last_error}")


def validate_output(
    payload: dict[str, Any],
    decision: datetime,
    documents: list[dict[str, Any]],
    schema: dict[str, Any],
) -> None:
    jsonschema.validate(instance=payload, schema=schema)
    if not isinstance(payload.get("events"), list):
        raise ValueError("GLM 输出缺少 events 数组")
    parsed = [Event.from_mapping(row) for row in payload["events"]]
    sources = {str(doc["source_id"]): doc for doc in documents}
    for event in parsed:
        invented = set(event.source_ids) - set(sources)
        if invented:
            raise ValueError(f"GLM 输出引用了不存在的 source_id: {sorted(invented)}")
        cited_times = {
            datetime.fromisoformat(sources[source_id]["published_at"])
            for source_id in event.source_ids
        }
        if event.published_at not in cited_times:
            raise ValueError(
                f"事件 {event.event_type} 的 published_at 与引用材料不一致")
    accepted = filter_point_in_time(parsed, decision)
    if len(accepted) != len(parsed):
        raise ValueError("GLM 输出包含未来事件或重复事件，已拒绝")


def normalize_output(
    payload: dict[str, Any], documents: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, int]]:
    """Repair only mechanical fields; drop rows requiring semantic invention."""
    sources = {str(doc["source_id"]): doc for doc in documents}
    stats = {"input": 0, "kept": 0, "dropped": 0, "repaired": 0}
    normalized: list[dict[str, Any]] = []
    for raw in payload.get("events", []) if isinstance(payload, dict) else []:
        stats["input"] += 1
        if not isinstance(raw, dict) or not REQUIRED_SEMANTIC.issubset(raw):
            stats["dropped"] += 1
            continue
        row = {key: raw.get(key) for key in EVENT_KEYS}
        cited = [str(x) for x in (row.get("source_ids") or []) if str(x) in sources]
        if not cited:
            stats["dropped"] += 1
            continue
        repaired = set(raw) != EVENT_KEYS or cited != row.get("source_ids")
        row["source_ids"] = cited[:3]
        repaired |= len(cited) > 3
        # Source timestamps/types are deterministic properties of evidence.
        primary = sources[cited[0]]
        if row.get("published_at") != primary["published_at"]:
            repaired = True
        row["published_at"] = primary["published_at"]
        source_types = {sources[source_id]["source_type"] for source_id in cited}
        if len(source_types) != 1:
            # A single event cannot truthfully have multiple source_type values.
            row["source_ids"] = [cited[0]]
            repaired = True
        row["source_type"] = sources[row["source_ids"][0]]["source_type"]
        evidence = row.get("evidence_text")
        if isinstance(evidence, str) and len(evidence) > 120:
            evidence = evidence[:120]
            repaired = True
        row["evidence_text"] = evidence
        for numeric_key in ("actual", "expected"):
            numeric_value = row.get(numeric_key)
            if numeric_value is not None and not isinstance(
                numeric_value, (int, float)
            ):
                numeric_value = None
                repaired = True
            row[numeric_key] = numeric_value
        try:
            for key in (
                "direction", "magnitude", "exposure", "novelty",
                "source_quality", "evidence_confidence",
            ):
                value = float(row[key])
                lower = -1.0 if key == "direction" else 0.0
                clipped = float(np.clip(value, lower, 1.0))
                repaired |= clipped != value
                row[key] = clipped
            horizon = int(row["horizon_days"])
            row["horizon_days"] = min(max(horizon, 1), 730)
            repaired |= row["horizon_days"] != horizon
        except (TypeError, ValueError):
            stats["dropped"] += 1
            continue
        normalized.append(row)
        stats["kept"] += 1
        stats["repaired"] += int(repaired)

    # Keep the highest-confidence copy of each semantic/evidence identity.
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for row in normalized:
        try:
            event = Event.from_mapping(row)
        except ValueError:
            stats["dropped"] += 1
            stats["kept"] -= 1
            continue
        previous = best.get(event.identity)
        if previous is None or event.evidence_confidence > previous[0]:
            best[event.identity] = (event.evidence_confidence, row)
    stats["dropped"] += len(normalized) - len(best)
    rows = [item[1] for item in best.values()]
    if len(rows) > 24:
        rows.sort(
            key=lambda row: float(row.get("evidence_confidence") or 0),
            reverse=True)
        stats["dropped"] += len(rows) - 24
        rows = rows[:24]
    stats["kept"] = len(rows)
    return {"events": rows}, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--decision-time", required=True)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--schema", type=Path,
        default=Path(__file__).resolve().parents[1] / "schemas/stock_events.schema.json")
    parser.add_argument(
        "--model", default=os.environ.get("GLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--max-input-chars", type=int, default=120_000,
        help="全文按字符数分批，避免单次请求超过上下文")
    parser.add_argument(
        "--max-document-chars", type=int, default=4_000,
        help="单份PDF/网页送入模型的最大相关证据字符数")
    args = parser.parse_args()

    load_local_env(Path(__file__).resolve().parents[1] / ".env.glm")
    api_key = os.environ.get("GLM_API_KEY")
    if not api_key:
        raise SystemExit("缺少 GLM_API_KEY 环境变量；API Key 不应写入代码或参数")

    decision = _parse_decision(args.decision_time)
    material_payload = json.loads(args.materials.read_text())
    documents = _validate_material_bundle(material_payload, decision)
    documents, compaction = compact_documents(
        documents, args.max_document_chars)
    schema = json.loads(args.schema.read_text())
    chunks: list[dict[str, Any]] = []
    for document in documents:
        content = document.get("content", "")
        if len(content) <= args.max_input_chars:
            chunks.append(document)
            continue
        for offset in range(0, len(content), args.max_input_chars):
            part = dict(document)
            part["content"] = content[offset:offset + args.max_input_chars]
            part["chunk"] = {
                "offset": offset,
                "total_chars": len(content),
            }
            chunks.append(part)

    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for document in chunks:
        size = len(json.dumps(document, ensure_ascii=False))
        if current and current_chars + size > args.max_input_chars:
            batches.append(current)
            current, current_chars = [], 0
        current.append(document)
        current_chars += size
    if current:
        batches.append(current)

    all_rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for batch_no, batch in enumerate(batches, 1):
        prompt = USER_PROMPT.format(
            ticker=args.ticker,
            decision_time=args.decision_time,
            schema=json.dumps(schema, ensure_ascii=False),
            materials=json.dumps({"documents": batch}, ensure_ascii=False),
        )
        result, metadata = call_glm(
            prompt, api_key=api_key, model=args.model, timeout=args.timeout,
            max_attempts=args.max_attempts)
        result, normalization = normalize_output(result, documents)
        validate_output(result, decision, documents, schema)
        all_rows.extend(result["events"])
        calls.append({"batch": batch_no, "normalization": normalization, **metadata})

    parsed = [Event.from_mapping(row) for row in all_rows]
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for row, event in zip(all_rows, parsed):
        previous = best.get(event.identity)
        if previous is None or event.evidence_confidence > previous[0]:
            best[event.identity] = (event.evidence_confidence, row)
    final_rows = [item[1] for item in best.values()]
    result, final_normalization = normalize_output(
        {"events": final_rows}, documents)
    validate_output(result, decision, documents, schema)
    metadata = {
        "model": args.model,
        "batch_count": len(batches),
        "document_count": len(documents),
        "document_chunk_count": len(chunks),
        "compaction": compaction,
        "final_normalization": final_normalization,
        "calls": calls,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"validated GLM events -> {args.output}")
    print(f"request metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
