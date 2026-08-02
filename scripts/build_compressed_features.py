"""Build the two compressed daily feature panels (Scheme A: 19-dim
cluster-representative subset; Scheme B: 17-dim PCA) for 2023-05 ~ 2026-05,
and write them as CSV under condition_diffusion/train_data/.

Requires prepare_descriptor_history.py and fit_compression_models.py to have
already been run (forward-filled descriptor history + clip bounds/PCA model).
"""
from __future__ import annotations

import pandas as pd

from feature_config import (
    PCA_MODEL_PATH, CLIP_BOUNDS_PATH, REQUEST_END, REQUEST_START,
    SCHEME_A_CSV, SCHEME_A_FEATURES, SCHEME_B_CSV, SCHEME_B_N_COMPONENTS,
    TRAIN_DATA_DIR,
)
from raw_snapshot import build_snapshot, trading_days


def main() -> None:
    TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    clip_bounds = pd.read_pickle(CLIP_BOUNDS_PATH)
    model = pd.read_pickle(PCA_MODEL_PATH)
    scaler, pca = model["scaler"], model["pca"]
    pc_cols = [f"pc_{i + 1}" for i in range(SCHEME_B_N_COMPONENTS)]

    days = trading_days(REQUEST_START, REQUEST_END)
    print(f"{len(days)} trading days to process", flush=True)

    SCHEME_A_CSV.unlink(missing_ok=True)
    SCHEME_B_CSV.unlink(missing_ok=True)
    first_write = True
    n_written = 0

    for i, date in enumerate(days):
        snap = build_snapshot(date, clip_bounds=clip_bounds)
        if snap is None or snap.empty:
            print(f"{date}: no data, skipped", flush=True)
            continue

        a_df = snap[SCHEME_A_FEATURES].reset_index()
        a_df.insert(0, "date", date)
        a_df.to_csv(SCHEME_A_CSV, mode="w" if first_write else "a",
                   header=first_write, index=False)

        Xs = scaler.transform(snap.to_numpy())
        pcs = pca.transform(Xs)
        b_df = pd.DataFrame(pcs, columns=pc_cols, index=snap.index).reset_index()
        b_df.insert(0, "date", date)
        b_df.to_csv(SCHEME_B_CSV, mode="w" if first_write else "a",
                   header=first_write, index=False)

        first_write = False
        n_written += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(days)} days done ({n_written} written so far)", flush=True)

    print(f"DONE. {n_written}/{len(days)} days written.", flush=True)
    print(f"Scheme A -> {SCHEME_A_CSV}", flush=True)
    print(f"Scheme B -> {SCHEME_B_CSV}", flush=True)


if __name__ == "__main__":
    main()
