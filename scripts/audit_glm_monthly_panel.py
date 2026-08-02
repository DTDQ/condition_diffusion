#!/usr/bin/env python3
"""Strictly audit the balanced monthly 48-d GLM feature panel."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path


def months_between(start: str, end: str) -> list[str]:
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    current = date(sy, sm, 1)
    result: list[str] = []
    while (current.year, current.month) <= (ey, em):
        result.append(f"{current.year:04d}-{current.month:02d}")
        next_month = current.month + 1
        current = date(
            current.year + (next_month > 12),
            1 if next_month > 12 else next_month,
            1,
        )
    return result


def load_universe(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is None or "stock_id" not in rows.fieldnames:
            raise ValueError(f"{path}: missing stock_id column")
        ids = [row["stock_id"] for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: universe is empty or contains duplicates")
    return set(ids)


def load_expected_dates(path: Path, start: str, end: str) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is None or "date" not in rows.fieldnames:
            raise ValueError(f"{path}: missing date column")
        dates = sorted(
            {
                row["date"]
                for row in rows
                if start <= row["date"][:7] <= end
            }
        )
    if not dates:
        raise ValueError(f"{path}: no dates in requested range")
    return dates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-month", default="2023-01")
    parser.add_argument("--to-month", default="2026-07")
    parser.add_argument(
        "--universe", type=Path,
        default=Path("train_data/universe_2023-01_2026-07.csv"),
    )
    parser.add_argument(
        "--scheme-a", type=Path,
        default=Path("train_data/features_schemeA_2023-01_2026-07.csv"),
    )
    parser.add_argument(
        "--glm-template", default="train_data/features_48d_{month}.csv"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("train_data/glm_panel_audit_2023-01_2026-07.json"),
    )
    args = parser.parse_args()

    months = months_between(args.from_month, args.to_month)
    universe = load_universe(args.universe)
    expected_dates = load_expected_dates(
        args.scheme_a, args.from_month, args.to_month
    )
    expected_date_set = set(expected_dates)
    expected_by_month = Counter(value[:7] for value in expected_dates)

    first_header: list[str] | None = None
    value_names: list[str] = []
    mask_names: list[str] = []
    mask_observed = [0] * 48
    total_rows = 0
    actual_dates: list[str] = []
    monthly_rows: dict[str, int] = {}

    for month in months:
        path = Path(args.glm_template.format(month=month))
        if not path.is_file():
            raise FileNotFoundError(path)
        per_date_ids: dict[str, set[str]] = {}
        rows_this_month = 0
        with path.open(encoding="utf-8", newline="") as handle:
            rows = csv.reader(handle)
            try:
                header = next(rows)
            except StopIteration:
                raise ValueError(f"{path}: empty CSV") from None
            if first_header is None:
                first_header = header
                if header[:2] != ["date", "stock_id"]:
                    raise ValueError(f"{path}: invalid key columns")
                value_names = [x for x in header[2:] if not x.endswith("__observed")]
                mask_names = [x for x in header[2:] if x.endswith("__observed")]
                if len(value_names) != 48 or len(mask_names) != 48:
                    raise ValueError(
                        f"{path}: expected 48 values + 48 masks, got "
                        f"{len(value_names)} + {len(mask_names)}"
                    )
            elif header != first_header:
                raise ValueError(f"{path}: header mismatch")

            value_columns = [header.index(name) for name in value_names]
            mask_columns = [header.index(name) for name in mask_names]
            for line_number, row in enumerate(rows, 2):
                if len(row) != len(header):
                    raise ValueError(f"{path}:{line_number}: malformed row")
                row_date, stock_id = row[0], row[1]
                if row_date[:7] != month or row_date not in expected_date_set:
                    raise ValueError(f"{path}:{line_number}: unexpected date {row_date}")
                if stock_id not in universe:
                    raise ValueError(f"{path}:{line_number}: unexpected stock {stock_id}")
                day_ids = per_date_ids.setdefault(row_date, set())
                if stock_id in day_ids:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate key {(row_date, stock_id)}"
                    )
                day_ids.add(stock_id)
                for column in value_columns:
                    try:
                        numeric = float(row[column])
                    except ValueError as error:
                        raise ValueError(
                            f"{path}:{line_number}: non-numeric {header[column]}"
                        ) from error
                    if not math.isfinite(numeric):
                        raise ValueError(
                            f"{path}:{line_number}: non-finite {header[column]}"
                        )
                for index, column in enumerate(mask_columns):
                    try:
                        numeric = float(row[column])
                    except ValueError as error:
                        raise ValueError(
                            f"{path}:{line_number}: non-numeric {header[column]}"
                        ) from error
                    if numeric not in (0.0, 1.0):
                        raise ValueError(
                            f"{path}:{line_number}: non-binary {header[column]}={numeric}"
                        )
                    mask_observed[index] += int(numeric)
                rows_this_month += 1

        month_dates = sorted(per_date_ids)
        if len(month_dates) != expected_by_month[month]:
            raise ValueError(
                f"{path}: {len(month_dates)} dates != expected {expected_by_month[month]}"
            )
        for row_date, ids in per_date_ids.items():
            if ids != universe:
                raise ValueError(
                    f"{path}: {row_date} universe mismatch; "
                    f"missing={len(universe - ids)} extra={len(ids - universe)}"
                )
        expected_rows = expected_by_month[month] * len(universe)
        if rows_this_month != expected_rows:
            raise ValueError(f"{path}: {rows_this_month} rows != {expected_rows}")
        monthly_rows[month] = rows_this_month
        total_rows += rows_this_month
        actual_dates.extend(month_dates)
        print(
            f"{month}: ok dates={len(month_dates)} rows={rows_this_month}",
            flush=True,
        )

    if actual_dates != expected_dates:
        raise ValueError("combined GLM dates do not exactly match Scheme-A dates")
    expected_total = len(expected_dates) * len(universe)
    if total_rows != expected_total:
        raise ValueError(f"total rows {total_rows} != expected {expected_total}")

    report = {
        "status": "ok",
        "months": len(months),
        "date_start": expected_dates[0],
        "date_end": expected_dates[-1],
        "trading_dates": len(expected_dates),
        "stocks": len(universe),
        "rows": total_rows,
        "columns": len(first_header or []),
        "glm_value_features": len(value_names),
        "observed_mask_features": len(mask_names),
        "monthly_rows": monthly_rows,
        "mask_observed_rate": {
            name: count / total_rows
            for name, count in zip(mask_names, mask_observed, strict=True)
        },
        "mask_observed_rate_overall": sum(mask_observed) / (total_rows * 48),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
