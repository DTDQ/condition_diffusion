"""Filter features_schemeA.csv / features_schemeB.csv down to a balanced
panel: only keep stock_ids present on every single trading day in the file
(the intersection across all dates), so every day has exactly the same set
of stocks. Overwrites the two CSVs in place (via a temp file + rename).
"""
from __future__ import annotations

import os
from collections import defaultdict

import pandas as pd

from feature_config import SCHEME_A_CSV, SCHEME_B_CSV

CHUNKSIZE = 1_000_000


def find_common_stocks(path) -> tuple[set[str], int]:
    date_count: dict[str, int] = defaultdict(int)
    n_days_seen: set[str] = set()
    for chunk in pd.read_csv(path, usecols=["date", "stock_id"], chunksize=CHUNKSIZE):
        n_days_seen.update(chunk["date"].unique())
        for sid, c in chunk["stock_id"].value_counts().items():
            date_count[sid] += c
    n_days = len(n_days_seen)
    common = {sid for sid, c in date_count.items() if c == n_days}
    return common, n_days


def filter_csv(path, common: set[str]) -> None:
    tmp_path = str(path) + ".tmp"
    first = True
    n_in, n_out = 0, 0
    for chunk in pd.read_csv(path, chunksize=CHUNKSIZE):
        n_in += len(chunk)
        kept = chunk[chunk["stock_id"].isin(common)]
        n_out += len(kept)
        kept.to_csv(tmp_path, mode="w" if first else "a", header=first, index=False)
        first = False
    os.replace(tmp_path, path)
    print(f"{path}: {n_in} -> {n_out} rows", flush=True)


def main() -> None:
    print("pass 1: finding stocks present on every day (scheme A)...", flush=True)
    common, n_days = find_common_stocks(SCHEME_A_CSV)
    print(f"{n_days} days total, {len(common)} stocks present on ALL days", flush=True)

    print("pass 2: filtering scheme A...", flush=True)
    filter_csv(SCHEME_A_CSV, common)
    print("pass 2: filtering scheme B...", flush=True)
    filter_csv(SCHEME_B_CSV, common)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
