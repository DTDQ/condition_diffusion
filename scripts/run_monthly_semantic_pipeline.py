#!/usr/bin/env python3
"""Resumable per-stock material + GLM extraction with a terminal progress bar."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def progress(done: int, total: int, ok: int, failed: int, ticker: str) -> None:
    width = 36
    ratio = done / total if total else 1
    filled = int(width * ratio)
    bar = "█" * filled + "·" * (width - filled)
    print(
        f"\r[{bar}] {done}/{total} {ratio:6.2%} "
        f"ok={ok} fail={failed} {ticker:<12}",
        end="\n" if done == total else "", flush=True)


def valid_json(path: Path, required_key: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get(required_key), list)


def write_empty_event(event: Path, model: str) -> None:
    """Record the honest no-evidence result without starting a GLM process."""
    event.write_text('{"events": []}\n', encoding="utf-8")
    metadata = {
        "model": model,
        "batch_count": 0,
        "document_count": 0,
        "document_chunk_count": 0,
        "calls": [],
        "no_evidence": True,
    }
    event.with_suffix(event.suffix + ".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def metadata_cost(path: Path, input_price: float, output_price: float) -> float:
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return 0.0
    total = 0.0
    for call in payload.get("calls", []):
        usage = call.get("usage") or {}
        prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        completion = usage.get(
            "completion_tokens", usage.get("output_tokens", 0)) or 0
        total += prompt / 1_000_000 * input_price
        total += completion / 1_000_000 * output_price
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-csv", type=Path, required=True)
    parser.add_argument("--month", default="2026-06")
    parser.add_argument("--materials-dir", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument(
        "--raw-dir", type=Path,
        help="已弃用：为节省磁盘与输入 token，不再下载或保存原始全文")
    parser.add_argument(
        "--model", default=os.environ.get("GLM_MODEL", "glm-4.7-flashx"))
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--limit", type=int, help="smoke test only")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-document-chars", type=int, default=4_000)
    parser.add_argument("--max-budget-yuan", type=float, default=100.0)
    parser.add_argument(
        "--budget-reserve-yuan", type=float, default=10.0,
        help="为未落usage的异常请求预留预算")
    parser.add_argument(
        "--allow-paid-model", action="store_true",
        help="默认拒绝收费模型，防止批处理突破预算")
    args = parser.parse_args()

    load_local_env(Path(__file__).resolve().parents[1] / ".env.glm")
    if not os.environ.get("GLM_API_KEY"):
        raise SystemExit("GLM_API_KEY 未配置；请在启动本程序的环境中设置")
    allowed = {"glm-4.7-flash", "glm-4.7-flashx"}
    if args.model not in allowed and not args.allow_paid_model:
        raise SystemExit(
            f"模型 {args.model} 不在预算白名单 {sorted(allowed)}；"
            "如确需收费模型请显式传 --allow-paid-model")
    prices = {
        "glm-4.7-flash": (0.0, 0.0),
        "glm-4.7-flashx": (0.5, 3.0),
    }
    if args.model not in prices:
        raise SystemExit("未知模型价格，无法执行预算保护")
    input_price, output_price = prices[args.model]
    tickers = sorted(pd.read_csv(
        args.universe_csv, usecols=["stock_id"])["stock_id"].dropna().unique())
    if args.shard_count < 1:
        raise SystemExit("--shard-count 必须 >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index 必须位于 [0, shard-count)")
    tickers = tickers[args.shard_index::args.shard_count]
    if args.limit:
        tickers = tickers[:args.limit]
    start = f"{args.month}-01T00:00:00+08:00"
    # Month end is known for the requested YYYY-MM case; event timestamps are
    # later filtered again per trading-day during 48-d aggregation.
    year, month = map(int, args.month.split("-"))
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    end = f"{args.month}-{last_day:02d}T23:59:59+08:00"
    args.materials_dir.mkdir(parents=True, exist_ok=True)
    args.events_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.events_dir / (
        f"_failures.worker{args.shard_index}.log"
        if args.shard_count > 1 else "_failures.log")

    ok = failed = 0
    session_spent = 0.0
    stop_at = max(args.max_budget_yuan - args.budget_reserve_yuan, 0.0)
    for i, ticker in enumerate(tickers, 1):
        material = args.materials_dir / f"{ticker}.json"
        event = args.events_dir / f"{ticker}.json"
        if valid_json(event, "events"):
            ok += 1
            progress(i, len(tickers), ok, failed, ticker)
            continue
        try:
            if not valid_json(material, "documents"):
                subprocess.run([
                    sys.executable, "scripts/fetch_astock_materials.py",
                    "--ticker", ticker, "--start-time", start,
                    "--decision-time", end, "--page-size", str(args.page_size),
                    "--output", str(material),
                ], check=True, stdout=subprocess.DEVNULL)
            material_payload = json.loads(material.read_text())
            if not material_payload["documents"]:
                write_empty_event(event, args.model)
            else:
                subprocess.run([
                    sys.executable, "scripts/extract_events_with_glm.py",
                    "--ticker", ticker, "--decision-time", end,
                    "--materials", str(material), "--output", str(event),
                    "--model", args.model,
                    "--max-document-chars", str(args.max_document_chars),
                ], check=True, stdout=subprocess.DEVNULL)
            session_spent += metadata_cost(
                event.with_suffix(event.suffix + ".meta.json"),
                input_price, output_price)
            ok += 1
        except Exception as exc:
            failed += 1
            with log_path.open("a") as handle:
                handle.write(f"{ticker}\t{type(exc).__name__}: {exc}\n")
        progress(i, len(tickers), ok, failed, ticker)
        if session_spent >= stop_at:
            print(
                f"\n预算保护停止：成功调用usage估算 ¥{session_spent:.2f}，"
                f"已达到预留后上限 ¥{stop_at:.2f}（总预算 ¥{args.max_budget_yuan:.2f}）",
                flush=True)
            break


if __name__ == "__main__":
    main()
