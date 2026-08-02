"""Build a forward-filled per-stock history of the 10 Barra style factors +
64 trading_descriptors factors, matching sample_construction.py's
`ffill()` step (prepare_data() forward-fills every merged column per
stock_id across the full date range before any per-day use -- several
trading_descriptors columns, e.g. spread_bias/down_list/up_list, are only
populated on a subset of days).

Loads BARRA_FACTORS_DIR + TRADING_DESCRIPTORS_DIR daily files over
[REQUEST_START - HISTORY_BUFFER_DAYS, REQUEST_END], concatenates to long
format, sorts by (stock_id, date), forward-fills per stock, and writes one
cache file: DESCRIPTOR_HISTORY_PATH (columns: stock_id, date, 10 style +
64 descriptor columns).
"""
from __future__ import annotations

import pandas as pd

from feature_config import (
    ANALYSIS_DIR, BARRA_FACTORS_DIR, DESCRIPTOR_FACTORS, REQUEST_END,
    REQUEST_START, STYLE_FACTORS, TRADING_DESCRIPTORS_DIR,
)
from raw_snapshot import session_offset, trading_days

HISTORY_BUFFER_SESSIONS = 90  # extra lookback so ffill has a seed value near REQUEST_START
DESCRIPTOR_HISTORY_PATH = ANALYSIS_DIR / "descriptor_history_ffilled.pkl"


def build(start: str = REQUEST_START, end: str = REQUEST_END) -> pd.DataFrame:
    requested_days = trading_days(start, end)
    if not requested_days:
        raise ValueError(f"no trading days in {start}..{end}")
    # ``start`` may be a weekend/month boundary.  Offset from the first actual
    # session so the forward-fill history still receives its full seed buffer.
    buffered_start = (
        session_offset(requested_days[0], HISTORY_BUFFER_SESSIONS) or start
    )
    days = trading_days(buffered_start, end)
    print(f"loading {len(days)} days of barra_factors + trading_descriptors "
          f"({buffered_start} .. {end})", flush=True)

    frames = []
    for i, date in enumerate(days):
        barra_path = BARRA_FACTORS_DIR / f"{date}.pkl"
        desc_path = TRADING_DESCRIPTORS_DIR / f"{date}.pkl"
        if not barra_path.exists() or not desc_path.exists():
            continue
        barra = pd.read_pickle(barra_path)[["stock_id"] + STYLE_FACTORS]
        desc_raw = pd.read_pickle(desc_path)
        cols = [c for c in DESCRIPTOR_FACTORS if c in desc_raw.columns]
        desc = desc_raw[["stock_id"] + cols]
        merged = barra.merge(desc, on="stock_id", how="outer")
        merged.insert(1, "date", date)
        frames.append(merged)
        if (i + 1) % 100 == 0:
            print(f"  loaded {i + 1}/{len(days)}", flush=True)

    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(["stock_id", "date"])
    fill_cols = [c for c in full.columns if c not in ("stock_id", "date")]
    full[fill_cols] = full.groupby("stock_id")[fill_cols].ffill()
    full = full[full.date >= start].reset_index(drop=True)
    print(f"final history shape {full.shape}", flush=True)

    DESCRIPTOR_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_pickle(DESCRIPTOR_HISTORY_PATH)
    print(f"saved -> {DESCRIPTOR_HISTORY_PATH}", flush=True)
    return full


if __name__ == "__main__":
    build()
