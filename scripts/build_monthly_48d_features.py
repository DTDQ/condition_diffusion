#!/usr/bin/env python3
"""Build daily 48-d condition rows for a CSV-defined A-share universe.

The script first creates deterministic hard features from local point-in-time
caches. If monthly per-stock event files exist, it also aggregates events known
by each decision time. Output is append-only by day and resumable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from four_layer_conditions import (  # noqa: E402
    COMPANY_FEATURES, FEATURE_NAMES, MACRO_FEATURES, META_FEATURES,
    SECTOR_FEATURES, Event, aggregate_events, filter_point_in_time,
    meta_features,
)

TZ = ZoneInfo("Asia/Shanghai")
INDUSTRY_COLS = [
    "zx_petro", "zx_coal", "zx_metals", "zx_power", "zx_steel",
    "zx_chemicals", "zx_construct_eng", "zx_construct_mat", "zx_light_man",
    "zx_machinery", "zx_electr_equip", "zx_defense", "zx_automobiles",
    "zx_retail", "zx_hotels_lei", "zx_household_dur", "zx_textile",
    "zx_medical", "zx_food_bev", "zx_agriculture", "zx_banks",
    "zx_non_bank_fin", "zx_real_estate", "zx_transportation",
    "zx_electronic_comp", "zx_communication", "zx_computers", "zx_media",
    "zx_compre", "zx_securites", "zx_insurance", "zx_diversified_fin",
]


def progress(current: int, total: int, label: str = "", width: int = 32) -> None:
    ratio = current / total if total else 1
    filled = int(width * ratio)
    bar = "█" * filled + "·" * (width - filled)
    print(f"\r[{bar}] {current:>3}/{total} {ratio:6.1%} {label:<12}",
          end="\n" if current == total else "", flush=True)


def trading_days(data_dir: Path, month: str) -> list[str]:
    cal = pd.read_pickle(data_dir / "trade_cal.pkl")
    dates = cal.loc[cal.is_trading_day.eq(1), "date"].astype(str)
    return sorted(d for d in dates if d.startswith(month))


def prior_files(directory: Path, date: str, count: int) -> list[Path]:
    return [p for p in sorted(directory.glob("*.pkl")) if p.stem <= date][-count:]


def safe_frame(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_pickle(path)
    if columns is not None:
        frame = frame[[c for c in columns if c in frame.columns]]
    return frame


def industry_labels(status: pd.DataFrame) -> pd.Series:
    cols = [c for c in INDUSTRY_COLS if c in status.columns]
    matrix = status[cols].fillna(0).to_numpy()
    has_industry = matrix.max(axis=1) > 0
    labels = np.asarray(cols, dtype=object)[matrix.argmax(axis=1)]
    labels[~has_industry] = None
    return pd.Series(labels, index=status.index, name="industry")


def robust_time_score(panel: pd.DataFrame, value_col: str) -> pd.Series:
    def score(group: pd.DataFrame) -> float:
        values = pd.to_numeric(group[value_col], errors="coerce").dropna().to_numpy()
        if len(values) < 21:
            return np.nan
        history, current = values[:-1], values[-1]
        median = np.median(history)
        mad = np.median(np.abs(history - median))
        scale = max(1.4826 * mad, 1e-12)
        return float(np.clip((current - median) / scale / 3, -1, 1))
    return panel.groupby("stock_id", sort=False).apply(score, include_groups=False)


@lru_cache(maxsize=None)
def load_events(events_dir: Path | None, ticker: str) -> list[Event]:
    if events_dir is None:
        return []
    candidates = [events_dir / f"{ticker}.json"]
    candidates.extend(sorted(events_dir.glob(f"{ticker}_*.json")))
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return []
    payload = json.loads(path.read_text())
    return [Event.from_mapping(row) for row in payload.get("events", [])]


def build_day(
    date: str,
    pool: set[str],
    data_dir: Path,
    events_dir: Path | None,
    decision_hour: int,
) -> pd.DataFrame:
    market_dir = data_dir / "stock_market"
    status_dir = data_dir / "stock_status"
    flow_dir = data_dir / "stock_money_flow"
    equity_dir = data_dir / "stock_equity"

    market = safe_frame(
        market_dir / f"{date}.pkl", ["stock_id", "pct_chg", "amt"])
    status = safe_frame(status_dir / f"{date}.pkl")
    eligible = status.loc[
        status.stock_id.isin(pool)
        & status.is_ST.eq(0) & status.is_new_stock.eq(0)
        & status.winda.eq(1)
    ].copy()
    eligible["industry"] = industry_labels(eligible)
    base = eligible[["stock_id", "industry"]].merge(
        market, on="stock_id", how="inner")
    base = base.loc[base.stock_id.str.match(r"^[036]\d{5}\.(SZ|SH)$", na=False)]
    stocks = base["stock_id"].tolist()

    observed: dict[str, pd.Series | float] = {}

    # Company fund flow: current value relative to the previous 20 sessions.
    flow_frames = []
    for path in prior_files(flow_dir, date, 21):
        frame = safe_frame(path, ["stock_id", "net_inflow_rate_value"])
        frame = frame.loc[frame.stock_id.isin(stocks)].copy()
        frame["_date"] = path.stem
        flow_frames.append(frame)
    if len(flow_frames) == 21:
        flow_panel = pd.concat(flow_frames, ignore_index=True).sort_values("_date")
        observed["fund_flow"] = robust_time_score(
            flow_panel, "net_inflow_rate_value").reindex(stocks)

    # Dilution pressure: compare float A shares over roughly one month.
    equity_paths = prior_files(equity_dir, date, 22)
    if len(equity_paths) >= 2:
        old = safe_frame(equity_paths[0], ["stock_id", "float_a_share"]).set_index("stock_id")
        new = safe_frame(equity_paths[-1], ["stock_id", "float_a_share"]).set_index("stock_id")
        ratio = new["float_a_share"].div(old["float_a_share"]).reindex(stocks)
        observed["dilution_pressure"] = -np.tanh((ratio - 1) * 20)

    # Build sector daily mean-return history using each day's PIT membership.
    sector_history: list[pd.Series] = []
    index_history: list[float] = []
    for market_path in prior_files(market_dir, date, 20):
        status_path = status_dir / market_path.name
        if not status_path.exists():
            continue
        m = safe_frame(market_path, ["stock_id", "pct_chg"])
        s = safe_frame(status_path)
        s["industry"] = industry_labels(s)
        joined = s[["stock_id", "industry"]].merge(m, on="stock_id", how="inner")
        joined["ret"] = pd.to_numeric(joined["pct_chg"], errors="coerce") / 100
        sector_history.append(joined.groupby("industry")["ret"].mean())
        idx = m.loc[m.stock_id.eq("000300.SH"), "pct_chg"]
        index_history.append(float(idx.iloc[-1]) / 100 if len(idx) else np.nan)

    current_industry = base.set_index("stock_id")["industry"]
    if sector_history:
        latest_sector = sector_history[-1]
        observed["sector_return_1d"] = (
            current_industry.map(latest_sector).reindex(stocks) / 0.03).clip(-1, 1)
        current_returns = pd.to_numeric(base["pct_chg"], errors="coerce") / 100
        breadth_by_industry = base.assign(
            breadth=np.sign(current_returns.to_numpy())
        ).groupby("industry")["breadth"].mean()
        observed["sector_breadth"] = current_industry.map(
            breadth_by_industry).reindex(stocks).clip(-1, 1)
    if len(sector_history) >= 5:
        hist5 = pd.concat(sector_history[-5:], axis=1)
        compound5 = (1 + hist5).prod(axis=1) - 1
        observed["sector_return_5d"] = (
            current_industry.map(compound5).reindex(stocks) / 0.10).clip(-1, 1)
    if len(sector_history) >= 20 and np.isfinite(index_history[-20:]).all():
        hist20 = pd.concat(sector_history[-20:], axis=1)
        sector20 = (1 + hist20).prod(axis=1) - 1
        market20 = np.prod(1 + np.asarray(index_history[-20:])) - 1
        relative = (sector20 - market20) / 0.20
        observed["sector_relative_20d"] = current_industry.map(
            relative).reindex(stocks).clip(-1, 1)

    # Market-wide state, shared by every stock that day.
    changes = pd.to_numeric(
        market.loc[market.stock_id.str.match(r"^[036]\d{5}\.(SZ|SH)$", na=False),
                   "pct_chg"], errors="coerce").dropna()
    idx = market.loc[market.stock_id.eq("000300.SH"), "pct_chg"]
    if len(idx) and pd.notna(idx.iloc[-1]):
        observed["market_return"] = float(np.clip(float(idx.iloc[-1]) / 3, -1, 1))
    if len(changes):
        observed["market_volatility"] = float(np.clip(changes.std() / 5, -1, 1))
        observed["market_breadth"] = float(
            np.clip(changes.gt(0).mean() - changes.lt(0).mean(), -1, 1))
        observed["limit_sentiment"] = float(np.clip(
            (changes.ge(9.5).mean() - changes.le(-9.5).mean()) * 20, -1, 1))

    n = len(stocks)
    values = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    masks = np.zeros((n, len(FEATURE_NAMES)), dtype=np.uint8)
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    for name, raw in observed.items():
        j = index[name]
        if np.isscalar(raw):
            values[:, j] = np.clip(float(raw), -1, 1)
            masks[:, j] = 1
        else:
            series = pd.Series(raw).reindex(stocks)
            ok = series.notna().to_numpy()
            values[ok, j] = np.clip(series[ok].astype(float), -1, 1)
            masks[ok, j] = 1

    # Monthly events are filtered independently at each daily decision time.
    if events_dir is not None:
        decision = datetime.fromisoformat(
            f"{date}T{decision_hour:02d}:00:00+08:00")
        for row, ticker in enumerate(stocks):
            events = filter_point_in_time(load_events(events_dir, ticker), decision)
            semantic = {**aggregate_events(events, decision), **meta_features(events)}
            for name, value in semantic.items():
                j = index[name]
                values[row, j] = np.clip(value, -1, 1)
                masks[row, j] = 1

    output = pd.DataFrame({"date": date, "stock_id": stocks})
    for j, name in enumerate(FEATURE_NAMES):
        output[name] = values[:, j]
        output[f"{name}__observed"] = masks[:, j]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-csv", type=Path, required=True)
    parser.add_argument("--month", default="2026-06")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--events-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-hour", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = pd.read_csv(args.universe_csv, usecols=["stock_id"])
    pool = set(source["stock_id"].dropna().astype(str).unique())
    days = trading_days(args.data_dir, args.month)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    completed: set[str] = set()
    if args.output.exists():
        completed = set(pd.read_csv(args.output, usecols=["date"])["date"].astype(str))

    pending = [day for day in days if day not in completed]
    print(
        f"universe={len(pool)} trading_days={len(days)} "
        f"completed={len(completed)} pending={len(pending)}", flush=True)
    first = not args.output.exists()
    for i, day in enumerate(pending, 1):
        frame = build_day(
            day, pool, args.data_dir, args.events_dir, args.decision_hour)
        frame.to_csv(
            args.output, mode="w" if first else "a", header=first, index=False)
        first = False
        progress(i, len(pending), f"{day} {len(frame)}")
    print(f"output -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
