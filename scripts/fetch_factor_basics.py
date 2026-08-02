#!/usr/bin/env python3
"""Restore trading_calendar and daily stock_status from the Factor DB.

The Factor DB connector is loaded from the unpacked quant-data skill.  Access
is read-only.  Daily frames are written through a temporary file and renamed,
so an interrupted download never leaves a valid-looking partial pickle.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any

import pandas as pd


def load_factor(skill_dir: Path) -> Any:
    path = skill_dir / "factor.py"
    spec = importlib.util.spec_from_file_location("quant_data_factor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Factor DB connector: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_pickle(frame: pd.DataFrame, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_pickle(temporary)
    os.replace(temporary, target)


def fetch_calendar(factor: Any, data_dir: Path, from_date: str, to_date: str) -> None:
    connection = factor.get_connection("market")
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM trading_calendar WHERE date >= %s AND date <= %s "
                "ORDER BY date",
                (from_date, to_date),
            )
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
    finally:
        connection.close()
    frame = pd.DataFrame.from_records(rows, columns=columns)
    if frame.empty:
        raise RuntimeError("Factor DB returned no trading_calendar rows")
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    atomic_pickle(frame, data_dir / "trade_cal.pkl")
    print(f"calendar: {len(frame)} rows -> {data_dir / 'trade_cal.pkl'}", flush=True)


def fetch_status(factor: Any, data_dir: Path, from_date: str, to_date: str) -> None:
    output_dir = data_dir / "stock_status"
    output_dir.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_pickle(data_dir / "trade_cal.pkl")
    days = calendar.loc[calendar["is_trading_day"].eq(1), "date"].astype(str).tolist()
    written = skipped = rows_seen = 0
    for index, date in enumerate(days, 1):
        target = output_dir / f"{date}.pkl"
        if target.exists():
            skipped += 1
            continue
        connection = factor.get_connection("market")
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM stock_status WHERE date = %s ORDER BY stock_id",
                    (date,),
                )
                columns = [column[0] for column in cursor.description]
                rows = cursor.fetchall()
        finally:
            connection.close()
        if not rows:
            raise RuntimeError(f"Factor DB returned no stock_status rows for {date}")
        frame = pd.DataFrame.from_records(rows, columns=columns)
        frame["date"] = pd.to_datetime(frame["date"])
        atomic_pickle(frame, target)
        rows_seen += len(frame)
        written += 1
        if written % 20 == 0 or index == len(days):
            print(
                f"stock_status: {index}/{len(days)} days, {written} written",
                flush=True,
            )
    if rows_seen == 0:
        raise RuntimeError("Factor DB returned no stock_status rows")
    print(
        f"stock_status: rows={rows_seen:,} written_days={written} "
        f"skipped_days={skipped}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skill-dir", type=Path, default=Path(".skills/quant-data")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--from-date", default="2025-04-01")
    parser.add_argument("--to-date", default="2026-05-31")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    factor = load_factor(args.skill_dir.resolve())
    fetch_calendar(factor, args.data_dir, args.from_date, args.to_date)
    fetch_status(factor, args.data_dir, args.from_date, args.to_date)


if __name__ == "__main__":
    main()
