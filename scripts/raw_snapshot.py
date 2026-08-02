"""Rebuild the 81-dim cross-sectional feature snapshot for one trading day,
from condition_diffusion/data raw caches only.

Definitions replicated from deep_learning_framework/codes/sample_construction.py
(read for reference, not imported/executed):
  - open/high/low/close/vwap: adj_factor-adjusted price, expressed as the ratio
    to the same (adjusted) price PRICE_LOOKBACK_SESSIONS trading days earlier
  - volume/amt: log(x_t) - log(x_{t-1}) (one-session log change)
  - 10 Barra style factors + 64 trading_descriptors factors: looked up from the
    forward-filled history built by prepare_descriptor_history.py (several
    descriptor columns, e.g. spread_bias/down_list/up_list, are only populated
    on a subset of days -- the original pipeline forward-fills every merged
    column per stock before use), then the 64 descriptors are clipped to
    global [0.1%, 99.9%] bounds and cross-sectionally robust-standardized
    ((x - median) / IQR) per day. Barra factors are used as-is (already
    cross-sectionally standardized by construction as regression exposures).
Universe: winda membership, excludes ST / new-listing stocks, A-share only
(stock_id starting with '0'/'3'/'6'), matching sample_construction.py's filter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from feature_config import (
    DESCRIPTOR_FACTORS, K_PROPS, PRICE_LOOKBACK_SESSIONS, STOCK_MARKET_DIR,
    STOCK_STATUS_DIR, TRADE_CAL_PATH, UNIVERSE,
)

_trade_cal_cache: list[str] | None = None
_descriptor_history_cache: pd.DataFrame | None = None
_descriptor_history_by_date: dict[str, pd.DataFrame] | None = None


def trading_days(start: str | None = None, end: str | None = None) -> list[str]:
    global _trade_cal_cache
    if _trade_cal_cache is None:
        cal = pd.read_pickle(TRADE_CAL_PATH)
        _trade_cal_cache = cal.loc[cal.is_trading_day.eq(1), "date"].astype(str).sort_values().tolist()
    days = _trade_cal_cache
    if start is not None:
        days = [d for d in days if d >= start]
    if end is not None:
        days = [d for d in days if d <= end]
    return days


def session_offset(date: str, offset: int) -> str | None:
    """Trading day `offset` sessions before `date` (offset > 0 looks back)."""
    all_days = trading_days()
    try:
        idx = all_days.index(date)
    except ValueError:
        return None
    j = idx - offset
    return all_days[j] if j >= 0 else None


def robust_standardize(s: pd.Series) -> pd.Series:
    median = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if iqr == 0:
        iqr = 1
    return (s - median) / iqr


def _universe_stock_ids(date: str) -> set[str] | None:
    path = STOCK_STATUS_DIR / f"{date}.pkl"
    if not path.exists():
        return None
    status = pd.read_pickle(path)
    status = status.query("is_ST == 0 and is_new_stock == 0")
    status = status[status[UNIVERSE] == 1]
    ids = status["stock_id"]
    ids = ids[ids.str[0].isin(["0", "3", "6"])]
    return set(ids)


def _adjusted_prices(date: str) -> pd.DataFrame | None:
    path = STOCK_MARKET_DIR / f"{date}.pkl"
    if not path.exists():
        return None
    df = pd.read_pickle(path)[["stock_id"] + K_PROPS + ["volume", "amt", "adj_factor"]].copy()
    for col in K_PROPS:
        df[col] = df[col] * df["adj_factor"]
    return df.drop(columns=["adj_factor"])


def load_descriptor_history() -> pd.DataFrame:
    """Load prepare_descriptor_history.py's cached, forward-filled
    barra + trading_descriptors panel, and index it by date for fast lookup."""
    global _descriptor_history_cache, _descriptor_history_by_date
    if _descriptor_history_cache is None:
        from prepare_descriptor_history import DESCRIPTOR_HISTORY_PATH
        if not DESCRIPTOR_HISTORY_PATH.exists():
            raise FileNotFoundError(
                f"{DESCRIPTOR_HISTORY_PATH} missing -- run prepare_descriptor_history.py first")
        _descriptor_history_cache = pd.read_pickle(DESCRIPTOR_HISTORY_PATH)
        _descriptor_history_by_date = {
            d: g.set_index("stock_id") for d, g in _descriptor_history_cache.groupby("date")
        }
    return _descriptor_history_cache


def _descriptors_for_date(date: str) -> pd.DataFrame | None:
    load_descriptor_history()
    return _descriptor_history_by_date.get(date)


def price_volume_frame(date: str) -> pd.DataFrame | None:
    """7 base price/volume columns for `date`, restricted to the trading universe."""
    universe = _universe_stock_ids(date)
    if not universe:
        return None
    prev1 = session_offset(date, 1)
    prev39 = session_offset(date, PRICE_LOOKBACK_SESSIONS)
    if prev1 is None or prev39 is None:
        return None

    px_today = _adjusted_prices(date)
    px_prev1 = _adjusted_prices(prev1)
    px_prev39 = _adjusted_prices(prev39)
    if px_today is None or px_prev1 is None or px_prev39 is None:
        return None

    merged = px_today.merge(px_prev39, on="stock_id", suffixes=("", "_prev39"))
    merged = merged.merge(px_prev1[["stock_id", "volume", "amt"]], on="stock_id", suffixes=("", "_prev1"))
    merged = merged[merged.stock_id.isin(universe)].set_index("stock_id")

    out = pd.DataFrame(index=merged.index)
    for col in K_PROPS:
        prev_col = f"{col}_prev39"
        out[col] = merged[col] / merged[prev_col].replace(0, pd.NA)
    for col in ["volume", "amt"]:
        prev_col = f"{col}_prev1"
        today_val = merged[col].where(merged[col] > 0)
        prev_val = merged[prev_col].where(merged[prev_col] > 0)
        out[col] = np.log(today_val) - np.log(prev_val)
    return out


def build_snapshot(date: str, clip_bounds: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame | None:
    """Return a DataFrame indexed by stock_id with the 81 raw feature columns
    for `date`, or None if required source files are missing."""
    pv = price_volume_frame(date)
    if pv is None or pv.empty:
        return None

    desc_hist = _descriptors_for_date(date)
    if desc_hist is None:
        return None
    style = desc_hist[[c for c in desc_hist.columns if c not in DESCRIPTOR_FACTORS]]
    desc = desc_hist[[c for c in DESCRIPTOR_FACTORS if c in desc_hist.columns]].copy()
    if clip_bounds is not None:
        for col, (lo, hi) in clip_bounds.items():
            if col in desc.columns:
                desc[col] = desc[col].clip(lower=lo, upper=hi)
    desc = desc.apply(robust_standardize, axis=0)
    # Some descriptors (est_num_diff, long-window volume/turnover stats) are
    # legitimately unavailable for a meaningful fraction of stocks (no analyst
    # coverage, or not enough trading history yet for a 12-month window) even
    # after per-stock forward-fill. Treat "unknown" as "typical" (0 in
    # already-standardized space) rather than dropping the stock outright.
    style = style.fillna(0.0)
    desc = desc.fillna(0.0)

    from feature_config import FEATURE_NAMES
    out = pv.join(style, how="inner").join(desc, how="inner")
    out = out.reindex(columns=FEATURE_NAMES)
    out = out.replace([np.inf, -np.inf], np.nan)
    # open/high/low/close/vwap/volume/amt NaN means genuinely-missing trading
    # data (e.g. halted session) -- those rows are dropped, not imputed.
    return out.dropna(subset=K_PROPS + ["volume", "amt"])
