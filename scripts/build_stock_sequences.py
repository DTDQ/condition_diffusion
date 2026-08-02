"""Convert the two daily feature panels into stock-indexed time series.

The input CSVs are aligned on ``(date, stock_id)``.  The output directory
contains NumPy arrays that can be memory-mapped without loading the complete
panel into RAM::

    values.npy       float32 [n_stocks, n_dates, n_features]
    observed.npy     bool    [n_stocks, n_dates, n_features]
    stock_ids.npy    unicode [n_stocks]
    dates.npy        unicode [n_dates]
    feature_names.npy unicode [n_features]
    metadata.json

Scheme A and Scheme B are concatenated along the feature dimension.  A stock
can then be selected with ``stock_to_index[stock_id]`` and produces exactly
one chronologically ordered time series.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from itertools import zip_longest
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


KEY_COLUMNS = ("date", "stock_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheme-a", type=Path, default=Path("train_data/features_schemeA.csv")
    )
    parser.add_argument(
        "--scheme-b", type=Path, default=Path("train_data/features_schemeB.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("train_data/stock_sequences")
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output directory"
    )
    parser.add_argument(
        "--progress-every", type=int, default=250_000, metavar="ROWS"
    )
    return parser.parse_args()


def _reader(path: Path) -> tuple[list[str], Iterator[list[str]]]:
    handle = path.open("r", encoding="utf-8", newline="")
    reader = csv.reader(handle)
    try:
        header = next(reader)
    except StopIteration:
        handle.close()
        raise ValueError(f"empty CSV: {path}") from None
    if header[:2] != list(KEY_COLUMNS):
        handle.close()
        raise ValueError(
            f"{path}: first columns must be {KEY_COLUMNS}, got {tuple(header[:2])}"
        )

    def rows() -> Iterator[list[str]]:
        try:
            yield from reader
        finally:
            handle.close()

    return header, rows()


def discover_axes(path: Path) -> tuple[list[str], list[str], int]:
    """Discover sorted axes and reject duplicate panel keys."""
    _, rows = _reader(path)
    stocks: set[str] = set()
    dates: set[str] = set()
    keys: set[tuple[str, str]] = set()
    row_count = 0
    for row_count, row in enumerate(rows, 1):
        if len(row) < 2:
            raise ValueError(f"{path}: malformed row {row_count + 1}")
        key = (row[0], row[1])
        if key in keys:
            raise ValueError(f"{path}: duplicate key {key}")
        keys.add(key)
        dates.add(row[0])
        stocks.add(row[1])
    return sorted(stocks), sorted(dates), row_count


def _parse_values(raw: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    values = np.empty(len(raw), dtype=np.float32)
    observed = np.ones(len(raw), dtype=np.bool_)
    for index, text in enumerate(raw):
        try:
            value = float(text)
        except ValueError:
            value = math.nan
        if not math.isfinite(value):
            value = 0.0
            observed[index] = False
        values[index] = value
    return values, observed


def build_sequences(
    scheme_a: Path,
    scheme_b: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    progress_every: int = 250_000,
) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"output directory already exists: {output_dir}; pass --overwrite"
            )
        shutil.rmtree(output_dir)

    stock_ids, dates, expected_rows = discover_axes(scheme_a)
    expected_panel_rows = len(stock_ids) * len(dates)
    if expected_rows != expected_panel_rows:
        raise ValueError(
            "Scheme A is not a balanced panel: "
            f"{expected_rows} rows != {len(stock_ids)} stocks * {len(dates)} dates"
        )

    header_a, rows_a = _reader(scheme_a)
    header_b, rows_b = _reader(scheme_b)
    features_a = header_a[2:]
    features_b = header_b[2:]
    duplicate_features = set(features_a) & set(features_b)
    if duplicate_features:
        raise ValueError(f"feature names occur in both schemes: {duplicate_features}")
    feature_names = features_a + features_b

    output_dir.mkdir(parents=True)
    partial_values = output_dir / "values.npy.partial"
    partial_observed = output_dir / "observed.npy.partial"
    shape = (len(stock_ids), len(dates), len(feature_names))
    values = np.lib.format.open_memmap(partial_values, mode="w+", dtype="float32", shape=shape)
    observed = np.lib.format.open_memmap(
        partial_observed, mode="w+", dtype="bool", shape=shape
    )
    observed[:] = False

    stock_to_index = {stock_id: i for i, stock_id in enumerate(stock_ids)}
    date_to_index = {date: i for i, date in enumerate(dates)}
    seen = np.zeros((len(stock_ids), len(dates)), dtype=np.bool_)

    try:
        count = 0
        for count, pair in enumerate(zip_longest(rows_a, rows_b), 1):
            row_a, row_b = pair
            if row_a is None or row_b is None:
                raise ValueError("Scheme A and Scheme B have different row counts")
            key_a, key_b = tuple(row_a[:2]), tuple(row_b[:2])
            if key_a != key_b:
                raise ValueError(
                    f"input key mismatch at data row {count}: {key_a} != {key_b}"
                )
            if len(row_a) != len(header_a) or len(row_b) != len(header_b):
                raise ValueError(f"malformed CSV row at data row {count}: {key_a}")

            stock_index = stock_to_index[key_a[1]]
            date_index = date_to_index[key_a[0]]
            if seen[stock_index, date_index]:
                raise ValueError(f"duplicate key while writing: {key_a}")
            seen[stock_index, date_index] = True

            a_values, a_observed = _parse_values(row_a[2:])
            b_values, b_observed = _parse_values(row_b[2:])
            split = len(features_a)
            values[stock_index, date_index, :split] = a_values
            values[stock_index, date_index, split:] = b_values
            observed[stock_index, date_index, :split] = a_observed
            observed[stock_index, date_index, split:] = b_observed

            if progress_every > 0 and count % progress_every == 0:
                print(f"processed {count:,}/{expected_rows:,} rows", flush=True)

        if count != expected_rows or not seen.all():
            raise ValueError(
                f"incomplete aligned panel: processed {count} of {expected_rows} rows"
            )
        values.flush()
        observed.flush()
        del values, observed
        values = observed = None

        os.replace(partial_values, output_dir / "values.npy")
        os.replace(partial_observed, output_dir / "observed.npy")
        np.save(output_dir / "stock_ids.npy", np.asarray(stock_ids))
        np.save(output_dir / "dates.npy", np.asarray(dates))
        np.save(output_dir / "feature_names.npy", np.asarray(feature_names))
        metadata = {
            "shape": list(shape),
            "dtype": "float32",
            "axis_order": ["stock", "time", "feature"],
            "scheme_a_features": features_a,
            "scheme_b_features": features_b,
            "date_start": dates[0],
            "date_end": dates[-1],
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except BaseException:
        if values is not None:
            del values
        if observed is not None:
            del observed
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    print(f"wrote {output_dir}: shape={shape}", flush=True)


def main() -> None:
    args = parse_args()
    build_sequences(
        args.scheme_a,
        args.scheme_b,
        args.output_dir,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
