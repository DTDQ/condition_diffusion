#!/usr/bin/env python3
"""Resumably recover monthly GLM events and 48-d feature CSVs."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path


def months_between(start: str, end: str) -> list[str]:
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    current = date(sy, sm, 1)
    result = []
    while (current.year, current.month) <= (ey, em):
        result.append(f"{current.year:04d}-{current.month:02d}")
        month = current.month + 1
        current = date(current.year + (month > 12), 1 if month > 12 else month, 1)
    return result


def valid_event_ids(events: Path) -> set[str]:
    result = set()
    for path in events.glob("*.json"):
        if path.name.endswith(".meta.json"):
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("events"), list):
            result.add(path.stem)
    return result


def universe_ids(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        return {row["stock_id"] for row in csv.DictReader(handle) if row["stock_id"]}


def run_event_shards(
    month: str,
    universe: Path,
    model: str,
    shards: int,
    log_dir: Path,
    *,
    round_number: int,
    total_budget: float,
) -> None:
    materials = Path("data/materials") / month
    events = Path("data/events") / month
    # The monthly worker can fetch missing material bundles itself.  Creating
    # the directory here keeps a brand-new month fully resumable.
    materials.mkdir(parents=True, exist_ok=True)
    events.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen[bytes], object, Path]] = []
    for shard in range(shards):
        log_path = log_dir / f"{month}.{model}.round{round_number}.shard{shard}.log"
        handle = log_path.open("ab")
        command = [
            sys.executable,
            "scripts/run_monthly_semantic_pipeline.py",
            "--universe-csv", str(universe),
            "--month", month,
            "--materials-dir", str(materials),
            "--events-dir", str(events),
            "--model", model,
            "--shard-count", str(shards),
            "--shard-index", str(shard),
            "--max-budget-yuan", str(total_budget / shards),
            "--budget-reserve-yuan", "0",
        ]
        if model == "glm-4.7-flashx":
            command.append("--allow-paid-model")
        processes.append((subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT), handle, log_path))
    failures = []
    for process, handle, log_path in processes:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append((log_path, return_code))
    if failures:
        raise RuntimeError(f"GLM shards failed for {month}: {failures}")


def run_month(
    month: str,
    universe: Path,
    shards: int,
    log_dir: Path,
    *,
    paid_round_budget: float,
    max_paid_rounds: int,
    skip_free: bool,
) -> None:
    events = Path("data/events") / month
    expected = universe_ids(universe)
    complete = valid_event_ids(events)
    if not skip_free:
        run_event_shards(
            month, universe, "glm-4.7-flash", shards, log_dir,
            round_number=1, total_budget=1_000_000,
        )
        complete = valid_event_ids(events)
        print(
            f"{month}: free pass complete={len(complete)}/{len(expected)} ",
            f"missing={len(expected - complete)}",
            flush=True,
        )
    else:
        print(
            f"{month}: skipping saturated free model; "
            f"complete={len(complete)}/{len(expected)}",
            flush=True,
        )
    paid_round = 0
    while expected - complete and paid_round < max_paid_rounds:
        paid_round += 1
        before = len(complete)
        run_event_shards(
            month, universe, "glm-4.7-flashx", shards, log_dir,
            round_number=paid_round, total_budget=paid_round_budget,
        )
        complete = valid_event_ids(events)
        print(
            f"{month}: paid round {paid_round} complete={len(complete)}/{len(expected)} ",
            f"missing={len(expected - complete)}",
            flush=True,
        )
        if len(complete) == before:
            print(
                f"{month}: paid round {paid_round} made no progress; "
                "continuing the configured retry budget",
                flush=True,
            )
    missing = expected - complete
    if missing:
        raise RuntimeError(
            f"{month}: {len(missing)} event files still missing after fallback"
        )

    output = Path("train_data") / f"features_48d_{month}.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_monthly_48d_features.py",
            "--universe-csv", str(universe),
            "--month", month,
            "--data-dir", "data",
            "--events-dir", str(events),
            "--output", str(output),
            "--overwrite",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-month", default="2025-05")
    parser.add_argument("--to-month", default="2026-05")
    parser.add_argument(
        "--universe-csv", type=Path, default=Path("train_data/features_schemeA.csv")
    )
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--paid-round-budget", type=float, default=100.0)
    parser.add_argument("--max-paid-rounds", type=int, default=20)
    parser.add_argument("--skip-free", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/glm_recovery"))
    args = parser.parse_args()
    if args.shards < 1:
        raise SystemExit("--shards must be >= 1")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    for month in months_between(args.from_month, args.to_month):
        print(f"recovering {month} with {args.shards} shards", flush=True)
        run_month(
            month, args.universe_csv, args.shards, args.log_dir,
            paid_round_budget=args.paid_round_budget,
            max_paid_rounds=args.max_paid_rounds,
            skip_free=args.skip_free,
        )
        print(f"completed {month}", flush=True)


if __name__ == "__main__":
    main()
