#!/usr/bin/env python3
"""Build stock-indexed Scheme-A + GLM-value + GLM-mask sequences."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np


KEYS = ("date", "stock_id")


def csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        try:
            return next(csv.reader(handle))
        except StopIteration:
            raise ValueError(f"empty CSV: {path}") from None


def parse_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        return 0.0
    return value if math.isfinite(value) else 0.0


def discover_glm(
    paths: list[Path],
) -> tuple[list[str], list[str], list[str], set[tuple[str, str]]]:
    if not paths:
        raise ValueError("no GLM CSV files found")
    first = csv_header(paths[0])
    if first[:2] != list(KEYS):
        raise ValueError(f"{paths[0]}: first columns must be {KEYS}")
    value_names = [name for name in first[2:] if not name.endswith("__observed")]
    mask_names = [name for name in first[2:] if name.endswith("__observed")]
    if len(value_names) != 48 or len(mask_names) != 48:
        raise ValueError(
            f"expected 48 GLM values and 48 masks, got {len(value_names)} and {len(mask_names)}"
        )
    dates: set[str] = set()
    stocks: set[str] = set()
    keys: set[tuple[str, str]] = set()
    for path in paths:
        header = csv_header(path)
        if header != first:
            raise ValueError(f"GLM header mismatch: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = csv.reader(handle)
            next(rows)
            for line_number, row in enumerate(rows, 2):
                if len(row) != len(header):
                    raise ValueError(f"{path}:{line_number}: malformed row")
                key = (row[0], row[1])
                if key in keys:
                    raise ValueError(f"duplicate GLM key across monthly files: {key}")
                keys.add(key)
                dates.add(row[0])
                stocks.add(row[1])
    return sorted(dates), sorted(stocks), value_names + mask_names, keys


def build(
    scheme_a: Path,
    glm_paths: list[Path],
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    scheme_header = csv_header(scheme_a)
    if scheme_header[:2] != list(KEYS):
        raise ValueError(f"{scheme_a}: first columns must be {KEYS}")
    scheme_features = scheme_header[2:]
    if len(scheme_features) != 19:
        raise ValueError(f"expected 19 Scheme-A features, got {len(scheme_features)}")

    dates, stock_ids, glm_features, glm_keys = discover_glm(glm_paths)
    expected = len(dates) * len(stock_ids)
    if len(glm_keys) != expected:
        raise ValueError(
            f"GLM data is not balanced: {len(glm_keys)} rows != "
            f"{len(dates)} dates * {len(stock_ids)} stocks"
        )

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    all_features = scheme_features + glm_features
    shape = (len(stock_ids), len(dates), len(all_features))
    partial = output_dir / "values.npy.partial"
    values = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=shape
    )
    values[:] = np.nan
    stock_index = {stock: index for index, stock in enumerate(stock_ids)}
    date_index = {date: index for index, date in enumerate(dates)}

    try:
        matched_a: set[tuple[str, str]] = set()
        with scheme_a.open(encoding="utf-8", newline="") as handle:
            rows = csv.reader(handle)
            next(rows)
            for line_number, row in enumerate(rows, 2):
                if len(row) != len(scheme_header):
                    raise ValueError(f"{scheme_a}:{line_number}: malformed row")
                key = (row[0], row[1])
                if key not in glm_keys:
                    continue
                if key in matched_a:
                    raise ValueError(f"duplicate Scheme-A key: {key}")
                matched_a.add(key)
                i, j = stock_index[row[1]], date_index[row[0]]
                values[i, j, : len(scheme_features)] = [
                    parse_float(text) for text in row[2:]
                ]
        missing_a = glm_keys - matched_a
        if missing_a:
            raise ValueError(
                f"Scheme A is missing {len(missing_a)} GLM keys; example={next(iter(missing_a))}"
            )

        offset = len(scheme_features)
        written_glm: set[tuple[str, str]] = set()
        for path in glm_paths:
            header = csv_header(path)
            glm_order = [
                header.index(name)
                for name in glm_features
            ]
            with path.open(encoding="utf-8", newline="") as handle:
                rows = csv.reader(handle)
                next(rows)
                for line_number, row in enumerate(rows, 2):
                    key = (row[0], row[1])
                    if key in written_glm:
                        raise ValueError(f"duplicate GLM key while writing: {key}")
                    written_glm.add(key)
                    i, j = stock_index[row[1]], date_index[row[0]]
                    values[i, j, offset:] = [
                        parse_float(row[column]) for column in glm_order
                    ]

        if written_glm != glm_keys or not np.isfinite(values).all():
            raise ValueError("output tensor is incomplete or contains non-finite values")
        values.flush()
        del values
        values = None
        os.replace(partial, output_dir / "values.npy")
        np.save(output_dir / "stock_ids.npy", np.asarray(stock_ids))
        np.save(output_dir / "dates.npy", np.asarray(dates))
        np.save(output_dir / "feature_names.npy", np.asarray(all_features))
        metadata = {
            "shape": list(shape),
            "dtype": "float32",
            "axis_order": ["stock", "time", "feature"],
            "feature_groups": {
                "scheme_a": [0, 19],
                "glm_values": [19, 67],
                "glm_observed_mask": [67, 115],
            },
            "date_start": dates[0],
            "date_end": dates[-1],
            "source_scheme_a": str(scheme_a),
            "source_glm": [str(path) for path in glm_paths],
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        if values is not None:
            del values
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    print(f"wrote {output_dir}: shape={shape}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheme-a", type=Path, default=Path("train_data/features_schemeA.csv")
    )
    parser.add_argument(
        "--glm-glob", default="train_data/features_48d_20??-??.csv"
    )
    parser.add_argument("--from-month", default="2025-05")
    parser.add_argument("--to-month", default="2026-05")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("train_data/scheme_a_glm_sequences"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    glm_paths = [
        path for path in sorted(Path().glob(args.glm_glob))
        if args.from_month
        <= path.stem.removeprefix("features_48d_")
        <= args.to_month
    ]
    build(args.scheme_a, glm_paths, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
