#!/usr/bin/env python3
"""Build an atomic Scheme-A CSV for an explicit trading-date range."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from feature_config import CLIP_BOUNDS_PATH, SCHEME_A_FEATURES
from raw_snapshot import build_snapshot, trading_days


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-date", default="2023-01-01")
    parser.add_argument("--to-date", default="2026-07-31")
    parser.add_argument(
        "--output", type=Path,
        default=Path("train_data/features_schemeA_2023-01_2026-07.unbalanced.csv"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    clip_bounds = pd.read_pickle(CLIP_BOUNDS_PATH)
    days = trading_days(args.from_date, args.to_date)
    written_days: list[str] = []
    first = True
    try:
        for index, date in enumerate(days, 1):
            snapshot = build_snapshot(date, clip_bounds=clip_bounds)
            if snapshot is None or snapshot.empty:
                raise RuntimeError(f"no Scheme-A snapshot for trading day {date}")
            frame = snapshot[SCHEME_A_FEATURES].reset_index()
            frame.insert(0, "date", date)
            frame.to_csv(
                temporary, mode="w" if first else "a", header=first, index=False
            )
            first = False
            written_days.append(date)
            if index == 1 or index % 25 == 0 or index == len(days):
                print(
                    f"[{index}/{len(days)}] {date} rows={len(frame)}",
                    flush=True,
                )
        if written_days != days:
            raise RuntimeError("written trading dates differ from requested calendar")
        os.replace(temporary, args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Scheme A -> {args.output}; days={len(days)}", flush=True)


if __name__ == "__main__":
    main()
