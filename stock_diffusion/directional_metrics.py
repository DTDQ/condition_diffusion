from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def confusion(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    valid = (prediction != 0) & (truth != 0)
    predicted_up, actual_up = prediction > 0, truth > 0
    tp = int(np.sum(valid & predicted_up & actual_up))
    tn = int(np.sum(valid & ~predicted_up & ~actual_up))
    fp = int(np.sum(valid & predicted_up & ~actual_up))
    fn = int(np.sum(valid & ~predicted_up & actual_up))
    count = tp + tn + fp + fn
    up_recall = tp / (tp + fn) if tp + fn else float("nan")
    down_recall = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "accuracy": (tp + tn) / count,
        "balanced_accuracy": (up_recall + down_recall) / 2,
        "up_recall": up_recall,
        "down_recall": down_recall,
        "actual_up_rate": (tp + fn) / count,
        "predicted_up_rate": (tp + fp) / count,
        "true_up": tp,
        "true_down": tn,
        "false_up": fp,
        "false_down": fn,
        "valid": count,
        "ties": int(truth.size - count),
        "accuracy_ties_as_wrong": (tp + tn) / truth.size,
    }


def evaluate(samples: np.ndarray, truth: np.ndarray, history: np.ndarray) -> dict:
    point = samples.mean(axis=0)
    origin = history[:, -1:, :]
    origin_result = confusion(point - origin, truth - origin)
    predicted_daily = np.diff(np.concatenate((origin, point), axis=1), axis=1)
    actual_daily = np.diff(np.concatenate((origin, truth), axis=1), axis=1)
    daily_result = confusion(predicted_daily, actual_daily)
    return {"from_origin_all_features": origin_result, "daily_all_features": daily_result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    suffix = f"_{args.suffix}" if args.suffix else ""
    samples = np.load(args.output_dir / f"forecast_samples_normalized{suffix}.npy")
    truth = np.load(args.output_dir / f"truth_normalized{suffix}.npy")
    history = np.load(args.output_dir / f"history_normalized{suffix}.npy")
    feature_names = np.load(args.data_dir / "feature_names.npy", allow_pickle=False)[:19]
    point, origin = samples.mean(axis=0), history[:, -1:, :]
    result = evaluate(samples, truth, history)
    feature_rows = {}
    for index, name in enumerate(feature_names):
        feature_rows[str(name)] = confusion(
            point[:, :, index] - origin[:, :, index],
            truth[:, :, index] - origin[:, :, index],
        )
    result["from_origin_by_feature"] = feature_rows
    vwap_index = list(feature_names).index("vwap")
    vwap_prediction = point[:, :, vwap_index] - origin[:, :, vwap_index]
    vwap_truth = truth[:, :, vwap_index] - origin[:, :, vwap_index]
    result["vwap_from_origin"] = confusion(vwap_prediction, vwap_truth)
    result["vwap_daily"] = confusion(
        np.diff(
            np.concatenate((origin[:, :, vwap_index], point[:, :, vwap_index]), axis=1),
            axis=1,
        ),
        np.diff(
            np.concatenate((origin[:, :, vwap_index], truth[:, :, vwap_index]), axis=1),
            axis=1,
        ),
    )
    result["vwap_from_origin_by_horizon"] = [
        {"forecast_day": day + 1, **confusion(vwap_prediction[:, day], vwap_truth[:, day])}
        for day in range(truth.shape[1])
    ]
    destination = args.output_dir / f"directional_metrics{suffix}.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["vwap_from_origin"], ensure_ascii=False, indent=2))
    print(f"saved -> {destination}")


if __name__ == "__main__":
    main()
