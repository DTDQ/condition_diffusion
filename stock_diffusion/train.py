from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .data import StockWindowDataset, fit_or_load_normalizer
from .diffusion import ConditionalDiffusionTS
from .model import ConditionalStockTransformer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def validate(model: ConditionalDiffusionTS, loader: DataLoader, device: torch.device, batches: int) -> float:
    model.eval()
    losses = []
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        batch = move_batch(batch, device)
        loss, _ = model(batch["future"], batch["history"], batch["condition"], batch["condition_mask"])
        losses.append(float(loss))
    model.train()
    return float(np.mean(losses))


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train conditional Diffusion-TS on Scheme A + GLM data")
    parser.add_argument("--config", default="Config/stock_condition_diffusion.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run two training steps and one validation batch")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed_everything(int(config["train"]["seed"]))
    device = resolve_device(config["train"]["device"])
    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    data_cfg = config["data"]
    normalizer = fit_or_load_normalizer(
        data_cfg["root"], output_dir / "normalization.npz", data_cfg["train_end"]
    )
    common = dict(
        data_dir=data_cfg["root"],
        normalizer=normalizer,
        history_length=data_cfg["history_length"],
        prediction_length=data_cfg["prediction_length"],
        train_end=data_cfg["train_end"],
        val_end=data_cfg["val_end"],
    )
    train_set = StockWindowDataset(split="train", **common)
    val_set = StockWindowDataset(split="val", **common)
    worker_count = 0 if args.smoke_test else config["train"]["num_workers"]
    loader_args = dict(
        batch_size=config["train"]["batch_size"],
        num_workers=worker_count,
        pin_memory=device.type == "cuda",
        persistent_workers=worker_count > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    model_cfg = config["model"]
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
    diffusion_cfg = config["diffusion"]
    model = ConditionalDiffusionTS(
        denoiser,
        timesteps=diffusion_cfg["timesteps"],
        sampling_timesteps=diffusion_cfg["sampling_timesteps"],
        loss_type=diffusion_cfg["loss_type"],
        fourier_weight=diffusion_cfg.get("fourier_weight"),
        ddim_eta=diffusion_cfg["ddim_eta"],
        target_mode=diffusion_cfg.get("target_mode", "level"),
        residual_scale=diffusion_cfg.get("residual_scale", 2.0),
        condition_mode=diffusion_cfg.get("condition_mode", "glm"),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"]
    )
    use_amp = bool(config["train"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])

    max_steps = 2 if args.smoke_test else int(config["train"]["max_steps"])
    val_every = 1 if args.smoke_test else int(config["train"]["validate_every"])
    save_every = int(config["train"]["save_every"])
    iterator = iter(train_loader)
    model.train()
    best_val_loss = float("inf")
    for step in range(start_step + 1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
        with context:
            loss, metrics = model(
                batch["future"], batch["history"], batch["condition"], batch["condition_mask"]
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), config["train"]["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % config["train"]["log_every"] == 0:
            print(
                f"step={step} loss={loss.item():.6f} time={metrics['time_loss'].item():.6f} "
                f"fft={metrics['frequency_loss'].item():.6f}", flush=True
            )
        if step % val_every == 0:
            val_loss = validate(model, val_loader, device, 1 if args.smoke_test else config["train"]["validation_batches"])
            print(f"step={step} val_loss={val_loss:.6f}", flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    output_dir / "checkpoint-best.pt",
                    {"step": step, "val_loss": val_loss, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "config": config},
                )
        if step % save_every == 0 or step == max_steps:
            save_checkpoint(
                output_dir / f"checkpoint-{step}.pt",
                {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "config": config},
            )

    print(f"Finished {max_steps} steps on {device}; checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
