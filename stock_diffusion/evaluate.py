from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import RobustNormalizer, StockWindowDataset
from .diffusion import ConditionalDiffusionTS
from .model import ConditionalStockTransformer
from .train import move_batch, resolve_device, seed_everything


def build_model(config: dict) -> ConditionalDiffusionTS:
    data_cfg, model_cfg, diffusion_cfg = config["data"], config["model"], config["diffusion"]
    denoiser = ConditionalStockTransformer(
        target_dim=19,
        history_length=data_cfg["history_length"],
        prediction_length=data_cfg["prediction_length"],
        model_dim=model_cfg["model_dim"],
        encoder_layers=model_cfg["encoder_layers"],
        decoder_layers=model_cfg["decoder_layers"],
        heads=model_cfg["heads"],
        mlp_ratio=model_cfg["mlp_ratio"],
        dropout=model_cfg["dropout"],
    )
    return ConditionalDiffusionTS(
        denoiser,
        timesteps=diffusion_cfg["timesteps"],
        sampling_timesteps=diffusion_cfg["sampling_timesteps"],
        loss_type=diffusion_cfg["loss_type"],
        fourier_weight=diffusion_cfg.get("fourier_weight"),
        ddim_eta=diffusion_cfg["ddim_eta"],
        target_mode=diffusion_cfg.get("target_mode", "level"),
        residual_scale=diffusion_cfg.get("residual_scale", 2.0),
        condition_mode=diffusion_cfg.get("condition_mode", "glm"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate probabilistic stock forecasts")
    parser.add_argument("--config", default="Config/stock_condition_diffusion.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffle-condition", action="store_true")
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError("At least two samples are required for probabilistic metrics")
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed_everything(config["train"]["seed"])
    device = resolve_device(config["train"]["device"])
    output_dir = Path(config["train"]["output_dir"])
    normalizer = RobustNormalizer.load(output_dir / "normalization.npz")
    data_cfg = config["data"]
    dataset = StockWindowDataset(
        data_dir=data_cfg["root"], split="test", normalizer=normalizer,
        history_length=data_cfg["history_length"], prediction_length=data_cfg["prediction_length"],
        train_end=data_cfg["train_end"], val_end=data_cfg["val_end"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    shuffled_conditions = shuffled_masks = None
    if args.shuffle_condition:
        all_conditions = torch.stack([dataset[index]["condition"] for index in range(len(dataset))])
        all_masks = torch.stack([dataset[index]["condition_mask"] for index in range(len(dataset))])
        generator = torch.Generator().manual_seed(2026)
        permutation = torch.randperm(len(dataset), generator=generator)
        shuffled_conditions = all_conditions[permutation]
        shuffled_masks = all_masks[permutation]
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    feature_names = np.load(Path(data_cfg["root"]) / "feature_names.npy", allow_pickle=False)[:19].tolist()
    center = torch.as_tensor(normalizer.center, device=device)
    scale = torch.as_tensor(normalizer.scale, device=device)
    sums = {name: 0.0 for name in ("abs", "sq", "baseline_abs", "baseline_sq", "crps", "covered", "width")}
    per_abs = torch.zeros(19, device=device)
    per_sq = torch.zeros(19, device=device)
    all_forecasts_normalized = []
    all_truth_normalized = []
    all_history_normalized = []
    count = 0
    row_offset = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if args.shuffle_condition:
                batch_size_cpu = batch["history"].shape[0]
                batch["condition"] = shuffled_conditions[row_offset : row_offset + batch_size_cpu]
                batch["condition_mask"] = shuffled_masks[row_offset : row_offset + batch_size_cpu]
                row_offset += batch_size_cpu
            batch = move_batch(batch, device)
            batch_size = batch["history"].shape[0]
            repeated_history = batch["history"].repeat_interleave(args.samples, dim=0)
            repeated_condition = batch["condition"].repeat_interleave(args.samples, dim=0)
            repeated_mask = batch["condition_mask"].repeat_interleave(args.samples, dim=0)
            parallel_forecasts = model.sample(repeated_history, repeated_condition, repeated_mask)
            forecasts = parallel_forecasts.reshape(
                batch_size, args.samples, *parallel_forecasts.shape[1:]
            ).transpose(0, 1)
            all_forecasts_normalized.append(forecasts.cpu().numpy())
            all_truth_normalized.append(batch["future"].cpu().numpy())
            all_history_normalized.append(batch["history"].cpu().numpy())
            truth = batch["future"] * scale + center
            forecasts = forecasts * scale + center
            point = forecasts.mean(dim=0)
            baseline = batch["history"][:, -1:, :] * scale + center
            baseline = baseline.expand_as(truth)
            error = point - truth
            baseline_error = baseline - truth
            elements = truth.numel()
            sums["abs"] += error.abs().sum().item()
            sums["sq"] += error.square().sum().item()
            sums["baseline_abs"] += baseline_error.abs().sum().item()
            sums["baseline_sq"] += baseline_error.square().sum().item()
            per_abs += error.abs().sum((0, 1))
            per_sq += error.square().sum((0, 1))
            first = (forecasts - truth.unsqueeze(0)).abs().mean(dim=0)
            pairwise = (forecasts[:, None] - forecasts[None, :]).abs().mean((0, 1))
            sums["crps"] += (first - 0.5 * pairwise).sum().item()
            lower = torch.quantile(forecasts, 0.1, dim=0)
            upper = torch.quantile(forecasts, 0.9, dim=0)
            sums["covered"] += ((truth >= lower) & (truth <= upper)).sum().item()
            sums["width"] += (upper - lower).sum().item()
            count += elements
            print(f"test batch {batch_index}/{len(loader)}", flush=True)

    observations_per_feature = len(dataset) * data_cfg["prediction_length"]
    mae, rmse = sums["abs"] / count, (sums["sq"] / count) ** 0.5
    baseline_mae = sums["baseline_abs"] / count
    future_starts = dataset.dates[dataset.starts + data_cfg["history_length"]]
    future_ends = dataset.dates[
        dataset.starts + data_cfg["history_length"] + data_cfg["prediction_length"] - 1
    ]
    result = {
        "checkpoint_step": checkpoint["step"],
        "test_period": [str(future_starts.min()), str(future_ends.max())],
        "stock_count": int(len(dataset.stock_ids)),
        "forecast_origins": int(len(dataset.starts)),
        "test_windows": len(dataset),
        "ensemble_samples": args.samples,
        "metrics_original_scale": {
            "mae": mae,
            "rmse": rmse,
            "crps": sums["crps"] / count,
            "picp_80": sums["covered"] / count,
            "mean_interval_width_80": sums["width"] / count,
            "persistence_mae": baseline_mae,
            "persistence_rmse": (sums["baseline_sq"] / count) ** 0.5,
            "mae_improvement_vs_persistence": 1 - mae / baseline_mae,
        },
        "per_feature": {
            name: {
                "mae": float(per_abs[index].item() / observations_per_feature),
                "rmse": float((per_sq[index].item() / observations_per_feature) ** 0.5),
            }
            for index, name in enumerate(feature_names)
        },
    }
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    destination = output_dir / f"test_metrics{suffix}.json"
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    np.save(output_dir / f"forecast_samples_normalized{suffix}.npy", np.concatenate(all_forecasts_normalized, axis=1))
    np.save(output_dir / f"truth_normalized{suffix}.npy", np.concatenate(all_truth_normalized, axis=0))
    np.save(output_dir / f"history_normalized{suffix}.npy", np.concatenate(all_history_normalized, axis=0))
    print(json.dumps(result["metrics_original_scale"], ensure_ascii=False, indent=2))
    print(f"Saved metrics to {destination}")


if __name__ == "__main__":
    main()
