#!/usr/bin/env python3
"""Fetch the minimal daily Factor-DB caches required by this project.

Rows are queried one table/month at a time and split into atomic per-day
pickle files.  Existing daily files are retained, making the command safe to
resume after interruption.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


STYLE_FACTORS = [
    "beta", "earnings_yield", "growth", "leverage", "liquidity",
    "momentum", "nlsize", "size", "value", "volatility",
]
DESCRIPTOR_FACTORS = [
    "amt_1m_3m", "swap_1m", "swap_3m", "swap_1y", "corr_close_turnover",
    "ivol", "ivr", "spread_bias", "est_num_diff", "illiq",
    "ln_volume_mean_1m", "ln_volume_mean_3m", "ln_volume_mean_6m", "ln_volume_mean_12m",
    "ln_volume_std_1m", "ln_volume_std_3m", "ln_volume_std_6m", "ln_volume_std_12m",
    "turnover_mean_1m", "turnover_mean_3m", "turnover_mean_6m",
    "turnover_std_1m", "turnover_std_3m", "turnover_std_6m",
    "turnover_stdrate_1m", "turnover_stdrate_3m", "turnover_stdrate_6m",
    "volume_1m_div_12m", "volume_1m_minus_12m", "volume_std_1m_div_12m",
    "clo_5d_60d", "close_max_div_min_1m", "close_max_div_min_3m", "close_max_div_min_6m",
    "vwap_5d_60d", "return_max_1m", "return_std_1m", "return_std_1w",
    "return_std_3m", "return_std_6m", "return_std_12m", "mom_1y",
    "ideal_reversal", "down_list", "mom_3m", "mom_6m", "mom_2y",
    "mom_1y_1m", "reverse_1m", "up_list", "duvol", "ncskew", "ana_cov",
    "specific_mom1", "specific_mom6", "specific_mom12", "io_to_float_a_share",
    "stk_quantity_g", "mean_stkvaluetonav", "delta_io_to_float_share",
    "top_ten_io_to_float_a_share", "top_ten_stk_quantity_g",
    "top_ten_mean_stkvaluetonav", "delta_top_ten_io",
]
STATUS_COLUMNS = [
    "stock_status_id", "date", "stock_id", "is_new_stock", "is_ST",
    "szzz", "sz50", "sz380", "hs300", "zz500", "zz800", "zz1000",
    "cyb", "cybz", "winda", "sz380_weight", "hs300_weight", "zz500_weight",
    "zz800_weight", "zz1000_weight", "cyb_weight", "cybz_weight",
    "zx_petro", "zx_coal", "zx_metals", "zx_power", "zx_steel",
    "zx_chemicals", "zx_construct_eng", "zx_construct_mat", "zx_light_man",
    "zx_machinery", "zx_electr_equip", "zx_defense", "zx_automobiles",
    "zx_retail", "zx_hotels_lei", "zx_household_dur", "zx_textile",
    "zx_medical", "zx_food_bev", "zx_agriculture", "zx_banks",
    "zx_non_bank_fin", "zx_real_estate", "zx_transportation",
    "zx_electronic_comp", "zx_communication", "zx_computers", "zx_media",
    "zx_compre", "zx_securites", "zx_insurance", "zx_diversified_fin",
]


@dataclass(frozen=True)
class TableSpec:
    database: str
    table: str
    directory: str
    columns: tuple[str, ...]


SPECS = {
    "stock_market": TableSpec(
        "market", "stock_market", "stock_market",
        tuple(["date", "stock_id", "open", "high", "low", "close", "volume",
               "amt", "pct_chg", "vwap", "adj_factor"]),
    ),
    "stock_status": TableSpec(
        "market", "stock_status", "stock_status", tuple(STATUS_COLUMNS)
    ),
    "stock_money_flow": TableSpec(
        "market", "stock_money_flow", "stock_money_flow",
        ("date", "stock_id", "net_inflow_rate_value"),
    ),
    "stock_equity": TableSpec(
        "market", "stock_equity", "stock_equity",
        ("date", "stock_id", "float_a_share"),
    ),
    "barra_factors": TableSpec(
        "factors", "barra_factors", "barra_factors",
        tuple(["date", "stock_id", *STYLE_FACTORS]),
    ),
    "trading_descriptors": TableSpec(
        "factors", "alpha_trading_descriptors", "trading_descriptors",
        tuple(["date", "stock_id", *DESCRIPTOR_FACTORS]),
    ),
}


def load_factor(skill_dir: Path) -> Any:
    path = skill_dir / "factor.py"
    spec = importlib.util.spec_from_file_location("quant_data_factor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Factor DB connector: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def month_ranges(start: str, end: str) -> list[tuple[str, str, str]]:
    start_month = pd.Timestamp(start).to_period("M")
    end_month = pd.Timestamp(end).to_period("M")
    result = []
    for period in pd.period_range(start_month, end_month, freq="M"):
        left = max(pd.Timestamp(start), period.start_time).date().isoformat()
        right = min(pd.Timestamp(end), period.end_time).date().isoformat()
        result.append((str(period), left, right))
    return result


def atomic_pickle(frame: pd.DataFrame, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_pickle(temporary)
    os.replace(temporary, target)


def fetch_month(
    factor: Any,
    spec: TableSpec,
    data_dir: Path,
    month: str,
    start: str,
    end: str,
    retries: int,
    expected_days: tuple[str, ...],
) -> tuple[str, str, int, int]:
    output_dir = data_dir / spec.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    if expected_days and all((output_dir / f"{day}.pkl").exists() for day in expected_days):
        return spec.directory, month, 0, len(expected_days)
    sql_columns = ", ".join(f"`{column}`" for column in spec.columns)
    sql = (
        f"SELECT {sql_columns} FROM `{spec.table}` "
        "WHERE `date` >= %s AND `date` <= %s ORDER BY `date`, `stock_id`"
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        connection = None
        try:
            connection = factor.get_connection(spec.database)
            with connection.cursor() as cursor:
                cursor.execute(sql, (start, end))
                columns = [item[0] for item in cursor.description]
                rows = cursor.fetchall()
            frame = pd.DataFrame.from_records(rows, columns=columns)
            if frame.empty:
                return spec.directory, month, 0, 0
            frame["date"] = pd.to_datetime(frame["date"])
            written = skipped = 0
            for date, daily in frame.groupby("date", sort=True):
                target = output_dir / f"{date:%Y-%m-%d}.pkl"
                if target.exists():
                    skipped += 1
                    continue
                daily = daily.reset_index(drop=True)
                if daily["stock_id"].duplicated().any():
                    raise RuntimeError(f"duplicate stock_id in {spec.table} on {date:%Y-%m-%d}")
                atomic_pickle(daily, target)
                written += 1
            return spec.directory, month, written, skipped
        except Exception as exc:  # network/server retries are expected on large ranges
            last_error = exc
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
        finally:
            if connection is not None:
                connection.close()
    raise RuntimeError(f"{spec.table} {month} failed after {retries} attempts: {last_error}")


def refresh_calendar(factor: Any, data_dir: Path, start: str, end: str) -> None:
    connection = factor.get_connection("market")
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM trading_calendar WHERE date >= %s AND date <= %s ORDER BY date",
                (start, end),
            )
            columns = [item[0] for item in cursor.description]
            rows = cursor.fetchall()
    finally:
        connection.close()
    frame = pd.DataFrame.from_records(rows, columns=columns)
    if frame.empty:
        raise RuntimeError("trading_calendar query returned no rows")
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    atomic_pickle(frame, data_dir / "trade_cal.pkl")
    print(f"calendar {frame.date.min()}..{frame.date.max()} rows={len(frame)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-dir", type=Path, default=Path(".skills/quant-data"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--from-date", default="2022-04-01")
    parser.add_argument("--to-date", default="2026-07-31")
    parser.add_argument("--tables", nargs="+", choices=sorted(SPECS), default=sorted(SPECS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    factor = load_factor(args.skill_dir.resolve())
    refresh_calendar(factor, args.data_dir, args.from_date, args.to_date)
    tasks = [
        (SPECS[name], month, start, end)
        for name in args.tables
        for month, start, end in month_ranges(args.from_date, args.to_date)
    ]
    calendar = pd.read_pickle(args.data_dir / "trade_cal.pkl")
    trading = calendar.loc[calendar["is_trading_day"].eq(1), "date"].astype(str)
    expected_by_month = {
        month: tuple(trading[trading.str.startswith(month)].tolist())
        for month, _, _ in month_ranges(args.from_date, args.to_date)
    }
    print(f"monthly table tasks={len(tasks)} workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                fetch_month, factor, spec, args.data_dir, month, start, end, args.retries,
                expected_by_month[month],
            ): (spec.directory, month)
            for spec, month, start, end in tasks
        }
        completed = 0
        for future in as_completed(futures):
            directory, month, written, skipped = future.result()
            completed += 1
            print(
                f"[{completed}/{len(tasks)}] {directory} {month}: "
                f"written_days={written} skipped_days={skipped}",
                flush=True,
            )


if __name__ == "__main__":
    main()
