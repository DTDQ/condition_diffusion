#!/usr/bin/env python3
"""Filter a Scheme-A CSV to stocks present exactly once on every date."""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 500_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts: dict[str, int] = defaultdict(int)
    dates: set[str] = set()
    rows_in = 0
    for chunk in pd.read_csv(
        args.input, usecols=["date", "stock_id"], chunksize=CHUNK_SIZE
    ):
        rows_in += len(chunk)
        dates.update(chunk["date"].astype(str).unique())
        for stock_id, count in chunk["stock_id"].value_counts().items():
            counts[str(stock_id)] += int(count)
    common = {stock_id for stock_id, count in counts.items() if count == len(dates)}
    if not common:
        raise RuntimeError("no stock is present on every date")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    first = True
    rows_out = 0
    try:
        for chunk in pd.read_csv(args.input, chunksize=CHUNK_SIZE):
            kept = chunk.loc[chunk["stock_id"].astype(str).isin(common)]
            rows_out += len(kept)
            kept.to_csv(
                temporary, mode="w" if first else "a", header=first, index=False
            )
            first = False
        expected = len(common) * len(dates)
        if rows_out != expected:
            raise RuntimeError(f"unbalanced result: {rows_out} rows != {expected}")
        os.replace(temporary, args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(
        f"balanced Scheme A: {rows_in:,} -> {rows_out:,} rows; "
        f"dates={len(dates)} stocks={len(common)} -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
