"""Fit the two pieces needed before the daily 81-dim -> compressed feature
build can run:
  1. Global [0.1%, 99.9%] clip bounds for the 64 trading_descriptors factors
     (sample_construction.py computes this once over the whole date range
     before any per-day standardization; we approximate it from a spread-out
     sample of days rather than the full panel, which is equivalent within
     sampling noise and far cheaper).
  2. A StandardScaler + PCA(17) fit on cross-sectionally-standardized 81-dim
     snapshots, for Scheme B. Fit once and reused for every day so that PCA
     component "3" means the same thing on every date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from feature_config import (
    ANALYSIS_DIR, CLIP_BOUNDS_PATH, DESCRIPTOR_FACTORS, FEATURE_NAMES,
    PCA_MODEL_PATH, REQUEST_END, REQUEST_START, SCHEME_B_N_COMPONENTS,
)
from raw_snapshot import _descriptors_for_date, build_snapshot, trading_days

SAMPLE_EVERY_N_DAYS = 12


def fit_clip_bounds(sample_dates: list[str]) -> dict[str, tuple[float, float]]:
    frames = []
    for d in sample_dates:
        desc_hist = _descriptors_for_date(d)
        if desc_hist is None:
            continue
        cols = [c for c in DESCRIPTOR_FACTORS if c in desc_hist.columns]
        frames.append(desc_hist[cols])
    pooled = pd.concat(frames, ignore_index=True)
    bounds = {}
    for col in DESCRIPTOR_FACTORS:
        if col not in pooled.columns:
            continue
        lo = pooled[col].quantile(0.001)
        hi = pooled[col].quantile(0.999)
        bounds[col] = (float(lo), float(hi))
    return bounds


def fit_pca(sample_dates: list[str], clip_bounds: dict[str, tuple[float, float]]):
    rows = []
    for d in sample_dates:
        snap = build_snapshot(d, clip_bounds=clip_bounds)
        if snap is not None and not snap.empty:
            rows.append(snap.to_numpy())
    X = np.concatenate(rows, axis=0)
    print(f"PCA fit sample matrix shape: {X.shape}", flush=True)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    pca = PCA(n_components=SCHEME_B_N_COMPONENTS).fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    print(f"top {SCHEME_B_N_COMPONENTS} PCs explain {cum[-1] * 100:.1f}% of variance", flush=True)
    return scaler, pca


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    all_days = trading_days(REQUEST_START, REQUEST_END)
    sample_dates = all_days[::SAMPLE_EVERY_N_DAYS]
    print(f"{len(all_days)} trading days total, sampling {len(sample_dates)} for fitting", flush=True)

    clip_bounds = fit_clip_bounds(sample_dates)
    pd.to_pickle(clip_bounds, CLIP_BOUNDS_PATH)
    print(f"saved clip bounds -> {CLIP_BOUNDS_PATH}", flush=True)

    scaler, pca = fit_pca(sample_dates, clip_bounds)
    pd.to_pickle({"scaler": scaler, "pca": pca, "feature_names": FEATURE_NAMES}, PCA_MODEL_PATH)
    print(f"saved PCA model -> {PCA_MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
