from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path("train_data/scheme_a_glm_sequences")
HISTORY_DIR = Path("outputs/ablation_history_only")
GLM_DIR = Path("outputs/ablation_history_glm")
REPORT_DIR = Path("outputs/ablation_report")


def crps_by_stock(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    first = np.abs(samples - truth[None]).mean(axis=0)
    pairwise = np.abs(samples[:, None] - samples[None, :]).mean(axis=(0, 1))
    return (first - 0.5 * pairwise).mean(axis=(1, 2))


def method_metrics(samples: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    point = samples.mean(axis=0)
    error = point - truth
    lower, upper = np.quantile(samples, (0.1, 0.9), axis=0)
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "crps": float(crps_by_stock(samples, truth).mean()),
        "picp_80": float(((truth >= lower) & (truth <= upper)).mean()),
        "interval_width_80": float((upper - lower).mean()),
    }


def paired_bootstrap(reference: np.ndarray, candidate: np.ndarray, seed: int = 42) -> dict[str, float]:
    """Positive difference means candidate has lower error than reference."""
    difference = reference - candidate
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(50):
        indices = rng.integers(0, len(difference), size=(100, len(difference)))
        values.append(difference[indices].mean(axis=1))
    bootstrap = np.concatenate(values)
    return {
        "mean_absolute_gain": float(difference.mean()),
        "relative_gain": float(difference.mean() / reference.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_probability_gain_positive": float((bootstrap > 0).mean()),
        "stock_win_rate": float((candidate < reference).mean()),
        "bootstrap_unit": "stock; conditional on the single May-2026 time window",
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    history_samples = np.load(HISTORY_DIR / "forecast_samples_normalized.npy")
    glm_samples = np.load(GLM_DIR / "forecast_samples_normalized.npy")
    shuffled_samples = np.load(GLM_DIR / "forecast_samples_normalized_shuffled_glm.npy")
    truth = np.load(HISTORY_DIR / "truth_normalized.npy")
    history = np.load(HISTORY_DIR / "history_normalized.npy")
    persistence = np.repeat(history[:, -1:, :], truth.shape[1], axis=1)[None]
    feature_names = np.load(ROOT / "feature_names.npy", allow_pickle=False)[:19]

    normalizer = np.load(HISTORY_DIR / "normalization.npz")
    center, scale = normalizer["center"], normalizer["scale"]
    dates = np.load(ROOT / "dates.npy", allow_pickle=False).astype("datetime64[D]")
    test_indices = np.flatnonzero(dates > np.datetime64("2026-04-30"))
    if len(test_indices) != truth.shape[1]:
        raise ValueError("Expected the test target to cover every available May date")
    raw = np.load(ROOT / "values.npy", mmap_mode="r")
    truth_raw = np.asarray(raw[:, test_indices, :19])
    origin_raw = np.asarray(raw[:, test_indices[0] - 1 : test_indices[0], :19])
    persistence_raw = np.repeat(origin_raw, len(test_indices), axis=1)[None]
    history_raw_samples = history_samples * scale + center
    glm_raw_samples = glm_samples * scale + center
    shuffled_raw_samples = shuffled_samples * scale + center

    normalized = {
        "persistence": method_metrics(persistence, truth),
        "history_only": method_metrics(history_samples, truth),
        "history_glm": method_metrics(glm_samples, truth),
        "glm_shuffled_at_test": method_metrics(shuffled_samples, truth),
    }
    raw_metrics = {
        "persistence": method_metrics(persistence_raw, truth_raw),
        "history_only": method_metrics(history_raw_samples, truth_raw),
        "history_glm": method_metrics(glm_raw_samples, truth_raw),
        "glm_shuffled_at_test": method_metrics(shuffled_raw_samples, truth_raw),
    }

    history_point, glm_point = history_samples.mean(0), glm_samples.mean(0)
    shuffled_point, persistence_point = shuffled_samples.mean(0), persistence[0]
    per_stock_history = np.abs(history_point - truth).mean(axis=(1, 2))
    per_stock_glm = np.abs(glm_point - truth).mean(axis=(1, 2))
    per_stock_shuffled = np.abs(shuffled_point - truth).mean(axis=(1, 2))
    bootstrap = {
        "glm_vs_history_mae": paired_bootstrap(per_stock_history, per_stock_glm, 42),
        "matched_glm_vs_shuffled_mae": paired_bootstrap(per_stock_shuffled, per_stock_glm, 43),
        "glm_vs_history_crps": paired_bootstrap(
            crps_by_stock(history_samples, truth), crps_by_stock(glm_samples, truth), 44
        ),
    }

    rows = []
    for index, name in enumerate(feature_names):
        values = {}
        for method, point in (
            ("persistence", persistence_point), ("history_only", history_point),
            ("history_glm", glm_point), ("glm_shuffled", shuffled_point),
        ):
            values[method] = float(np.abs(point[:, :, index] - truth[:, :, index]).mean())
        rows.append({
            "feature": str(name), **values,
            "glm_gain_vs_history": values["history_only"] - values["history_glm"],
            "glm_relative_gain_vs_history": (
                values["history_only"] - values["history_glm"]
            ) / values["history_only"],
            "matched_glm_gain_vs_shuffle": values["glm_shuffled"] - values["history_glm"],
        })
    feature_frame = pd.DataFrame(rows)
    feature_frame.to_csv(REPORT_DIR / "per_feature_metrics.csv", index=False)

    horizon_rows = []
    for day in range(truth.shape[1]):
        horizon_rows.append({
            "forecast_day": day + 1,
            "persistence": float(np.abs(persistence_point[:, day] - truth[:, day]).mean()),
            "history_only": float(np.abs(history_point[:, day] - truth[:, day]).mean()),
            "history_glm": float(np.abs(glm_point[:, day] - truth[:, day]).mean()),
            "glm_shuffled": float(np.abs(shuffled_point[:, day] - truth[:, day]).mean()),
        })
    horizon_frame = pd.DataFrame(horizon_rows)
    horizon_frame.to_csv(REPORT_DIR / "per_horizon_metrics.csv", index=False)

    history_checkpoint = torch.load(HISTORY_DIR / "checkpoint-best.pt", map_location="cpu", weights_only=False)
    glm_checkpoint = torch.load(GLM_DIR / "checkpoint-best.pt", map_location="cpu", weights_only=False)
    result = {
        "experiment": {
            "test_period": [str(dates[test_indices[0]]), str(dates[test_indices[-1]])],
            "stocks": int(len(truth)), "forecast_days": int(truth.shape[1]),
            "ensemble_samples": int(history_samples.shape[0]), "seed": 42,
            "history_only_best_step": int(history_checkpoint["step"]),
            "history_only_val_loss": float(history_checkpoint["val_loss"]),
            "history_glm_best_step": int(glm_checkpoint["step"]),
            "history_glm_val_loss": float(glm_checkpoint["val_loss"]),
        },
        "normalized_metrics": normalized,
        "unclipped_raw_metrics": raw_metrics,
        "paired_bootstrap": bootstrap,
        "feature_summary": {
            "glm_better_than_history_count": int((feature_frame.history_glm < feature_frame.history_only).sum()),
            "matched_glm_better_than_shuffled_count": int((feature_frame.history_glm < feature_frame.glm_shuffled).sum()),
        },
        "interpretation_limit": (
            "Stock bootstrap measures cross-sectional evidence within one May window; "
            "it does not establish robustness across months or market regimes."
        ),
    }
    (REPORT_DIR / "ablation_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    methods = ["persistence", "history_only", "history_glm", "glm_shuffled_at_test"]
    labels = ["Persistence", "History only", "History + GLM", "GLM shuffled"]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, [normalized[name]["mae"] for name in methods])
    plt.ylabel("Normalized MAE"); plt.title("May-2026 ablation")
    plt.xticks(rotation=12); plt.tight_layout()
    plt.savefig(REPORT_DIR / "overall_mae.png", dpi=180); plt.close()

    ordered = feature_frame.sort_values("glm_relative_gain_vs_history")
    colors = ["tab:green" if value > 0 else "tab:red" for value in ordered.glm_relative_gain_vs_history]
    plt.figure(figsize=(10, 7))
    plt.barh(ordered.feature, 100 * ordered.glm_relative_gain_vs_history, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("GLM MAE improvement over history-only (%)")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "glm_gain_by_feature.png", dpi=180); plt.close()

    plt.figure(figsize=(9, 5))
    for column, label in zip(("persistence", "history_only", "history_glm"), labels[:3]):
        plt.plot(horizon_frame.forecast_day, horizon_frame[column], marker="o", label=label)
    plt.xlabel("Forecast trading day"); plt.ylabel("Normalized MAE")
    plt.xticks(horizon_frame.forecast_day); plt.legend(); plt.tight_layout()
    plt.savefig(REPORT_DIR / "ablation_by_horizon.png", dpi=180); plt.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
