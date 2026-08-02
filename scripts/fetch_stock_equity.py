#!/usr/bin/env python3
"""Fetch daily market.stock_equity snapshots from the read-only Factor DB."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import pandas as pd


def load_factor(skill_dir: Path):
    spec = importlib.util.spec_from_file_location("quant_data_factor", skill_dir / "factor.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load factor.py from {skill_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/stock_equity"))
    parser.add_argument("--from-date", default="2023-05-01")
    parser.add_argument("--to-date", default="2026-05-31")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    factor = load_factor(args.skill_dir.resolve())
    conn = factor.get_connection("market")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM stock_equity
                WHERE date >= %s AND date <= %s
                ORDER BY date, stock_id
                """,
                (args.from_date, args.to_date),
            )
            columns = [item[0] for item in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()

    frame = pd.DataFrame.from_records(rows, columns=columns)
    if frame.empty:
        raise RuntimeError("Factor DB returned no stock_equity rows")
    frame["date"] = pd.to_datetime(frame["date"])

    written = 0
    skipped = 0
    for date, daily in frame.groupby("date", sort=True):
        target = output_dir / f"{date:%Y-%m-%d}.pkl"
        if target.exists():
            skipped += 1
            continue
        daily = daily.reset_index(drop=True)
        temporary = target.with_suffix(".pkl.tmp")
        daily.to_pickle(temporary)
        os.replace(temporary, target)
        written += 1

    print(
        f"rows={len(frame)} dates={frame['date'].nunique()} "
        f"written={written} skipped_existing={skipped}"
    )


if __name__ == "__main__":
    main()
