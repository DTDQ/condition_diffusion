from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn

from .data import RobustNormalizer

_UPSTREAM = Path(__file__).resolve().parents[1] / "third_party" / "Diffusion-TS"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

from Models.ts2vec.ts2vec import TS2Vec  # noqa: E402
from Utils.cross_correlation import CrossCorrelLoss  # noqa: E402


class SequenceDiscriminator(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.gru = nn.GRU(features, max(features // 2, 1), batch_first=True)
        self.output = nn.Linear(max(features // 2, 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.output(hidden[-1]).squeeze(-1)


class OneStepPredictor(nn.Module):
    def __init__(self, input_features: int, full_features: int) -> None:
        super().__init__()
        hidden = max(full_features // 2, 1)
        self.gru = nn.GRU(input_features, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(x)
        return torch.sigmoid(self.output(sequence))


def discriminative_score(
    real: torch.Tensor, fake: torch.Tensor, device: torch.device, seed: int
) -> tuple[float, float, float]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real_order = torch.randperm(len(real), generator=generator)
    fake_order = torch.randperm(len(fake), generator=generator)
    split = int(len(real) * 0.8)
    train_real, test_real = real[real_order[:split]], real[real_order[split:]]
    train_fake, test_fake = fake[fake_order[:split]], fake[fake_order[split:]]
    model = SequenceDiscriminator(real.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for iteration in range(2000):
        real_idx = torch.randint(len(train_real), (128,))
        fake_idx = torch.randint(len(train_fake), (128,))
        real_batch = train_real[real_idx].to(device)
        fake_batch = train_fake[fake_idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(real_batch), torch.ones(128, device=device))
        loss = loss + loss_fn(model(fake_batch), torch.zeros(128, device=device))
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        real_pred = model(test_real.to(device)) > 0
        fake_pred = model(test_fake.to(device)) > 0
    real_acc = real_pred.float().mean().item()
    fake_acc = (~fake_pred).float().mean().item()
    accuracy = 0.5 * (real_acc + fake_acc)
    return abs(accuracy - 0.5), fake_acc, real_acc


def predictive_score(
    real: torch.Tensor, fake: torch.Tensor, device: torch.device, seed: int
) -> float:
    torch.manual_seed(seed)
    input_features = real.shape[-1] - 1
    model = OneStepPredictor(input_features, real.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    for iteration in range(5000):
        indices = torch.randint(len(fake), (128,))
        batch = fake[indices].to(device)
        inputs = batch[:, :-1, :input_features]
        target = batch[:, 1:, -1:]
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs) - target).abs().mean()
        loss.backward()
        optimizer.step()
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(real), 512):
            batch = real[start : start + 512].to(device)
            target = batch[:, 1:, -1:]
            total += (model(batch[:, :-1, :input_features]) - target).abs().sum().item()
            count += target.numel()
    return total / count


def context_fid(real: np.ndarray, fake: np.ndarray, device: torch.device, seed: int) -> float:
    np.random.seed(seed)
    torch.manual_seed(seed)
    encoder = TS2Vec(
        input_dims=real.shape[-1], device=device, batch_size=64, lr=0.001,
        output_dims=320, max_train_length=3000,
    )
    encoder.fit(real, verbose=False)
    real_repr = encoder.encode(real, encoding_window="full_series", batch_size=256)
    fake_repr = encoder.encode(fake, encoding_window="full_series", batch_size=256)
    mean_real, mean_fake = real_repr.mean(0), fake_repr.mean(0)
    cov_real, cov_fake = np.cov(real_repr, rowvar=False), np.cov(fake_repr, rowvar=False)
    covariance_mean = sqrtm(cov_real.dot(cov_fake))
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    return float(np.square(mean_real - mean_fake).sum() + np.trace(
        cov_real + cov_fake - 2 * covariance_mean
    ))


def summary(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)), "runs": values}


def save_plots(
    plot_dir: Path,
    real: np.ndarray,
    fake: np.ndarray,
    samples: np.ndarray,
    history: np.ndarray,
    normalizer: RobustNormalizer,
    feature_names: list[str],
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    selected = rng.choice(len(real), min(2000, len(real)), replace=False)
    real_prepared = real[selected].mean(axis=2)
    fake_prepared = fake[selected].mean(axis=2)

    pca = PCA(n_components=2).fit(real_prepared)
    real_pca, fake_pca = pca.transform(real_prepared), pca.transform(fake_prepared)
    plt.figure(figsize=(8, 6))
    plt.scatter(real_pca[:, 0], real_pca[:, 1], s=7, alpha=0.25, label="Real")
    plt.scatter(fake_pca[:, 0], fake_pca[:, 1], s=7, alpha=0.25, label="Generated")
    plt.legend(); plt.title("PCA: real vs generated test trajectories"); plt.tight_layout()
    plt.savefig(plot_dir / "pca_real_vs_generated.png", dpi=180); plt.close()

    combined = np.concatenate((real_prepared, fake_prepared))
    embedding = TSNE(
        n_components=2, perplexity=40, max_iter=1000, init="pca",
        learning_rate="auto", random_state=42,
    ).fit_transform(combined)
    half = len(real_prepared)
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:half, 0], embedding[:half, 1], s=7, alpha=0.25, label="Real")
    plt.scatter(embedding[half:, 0], embedding[half:, 1], s=7, alpha=0.25, label="Generated")
    plt.legend(); plt.title("t-SNE: real vs generated test trajectories"); plt.tight_layout()
    plt.savefig(plot_dir / "tsne_real_vs_generated.png", dpi=180); plt.close()

    plt.figure(figsize=(8, 5))
    sns.kdeplot(real_prepared.reshape(-1), label="Real", linewidth=2)
    sns.kdeplot(fake_prepared.reshape(-1), label="Generated", linewidth=2, linestyle="--")
    plt.legend(); plt.xlabel("Mean normalized feature value"); plt.title("Kernel density")
    plt.tight_layout(); plt.savefig(plot_dir / "kernel_density.png", dpi=180); plt.close()

    real_flat, fake_flat = real.reshape(-1, real.shape[-1]), fake.reshape(-1, fake.shape[-1])
    real_corr, fake_corr = np.corrcoef(real_flat, rowvar=False), np.corrcoef(fake_flat, rowvar=False)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for axis, matrix, title, limit in zip(
        axes, (real_corr, fake_corr, fake_corr - real_corr),
        ("Real correlation", "Generated correlation", "Difference"), (1, 1, 0.5),
    ):
        sns.heatmap(matrix, ax=axis, cmap="coolwarm", vmin=-limit, vmax=limit, cbar=False)
        axis.set_title(title)
    fig.tight_layout(); fig.savefig(plot_dir / "correlation_heatmaps.png", dpi=180); plt.close(fig)

    real_original = normalizer.inverse(real)
    samples_original = normalizer.inverse(samples)
    history_original = normalizer.inverse(history)
    point = samples_original.mean(axis=0)
    lower, upper = np.quantile(samples_original, (0.1, 0.9), axis=0)
    figure, axes = plt.subplots(5, 4, figsize=(18, 18), sharex=True)
    x_history = np.arange(-history.shape[1], 0)
    x_future = np.arange(real.shape[1])
    for feature, axis in enumerate(axes.flat):
        if feature >= len(feature_names):
            axis.axis("off"); continue
        axis.plot(x_history, history_original[0, :, feature], color="0.65", linewidth=1, label="History")
        axis.plot(x_future, real_original[0, :, feature], color="black", linewidth=1.5, label="Truth")
        axis.plot(x_future, point[0, :, feature], color="tab:blue", linewidth=1.2, label="Forecast")
        axis.fill_between(x_future, lower[0, :, feature], upper[0, :, feature], color="tab:blue", alpha=0.2)
        axis.set_title(feature_names[feature], fontsize=9)
    axes[0, 0].legend(fontsize=7); figure.suptitle("Example stock: history, test truth, forecast and 80% interval")
    figure.tight_layout(); figure.savefig(plot_dir / "trajectory_all_features.png", dpi=180); plt.close(figure)

    point_normalized = samples.mean(axis=0)
    persistence = np.repeat(history[:, -1:, :], real.shape[1], axis=1)
    model_mae = np.abs(point_normalized - real).mean(axis=(0, 2))
    baseline_mae = np.abs(persistence - real).mean(axis=(0, 2))
    plt.figure(figsize=(9, 5)); days = np.arange(1, real.shape[1] + 1)
    plt.plot(days, model_mae, marker="o", label="Diffusion")
    plt.plot(days, baseline_mae, marker="o", label="Persistence")
    plt.xlabel("Forecast trading day"); plt.ylabel("Normalized MAE"); plt.xticks(days)
    plt.legend(); plt.title("MAE by forecast horizon"); plt.tight_layout()
    plt.savefig(plot_dir / "mae_by_horizon.png", dpi=180); plt.close()

    nominal, empirical = [], []
    for level in np.arange(0.1, 1.0, 0.1):
        tail = (1 - level) / 2
        lo, hi = np.quantile(samples, (tail, 1 - tail), axis=0)
        nominal.append(level); empirical.append(((real >= lo) & (real <= hi)).mean())
    plt.figure(figsize=(6, 6)); plt.plot(nominal, empirical, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="black", label="Ideal")
    plt.xlabel("Nominal coverage"); plt.ylabel("Empirical coverage"); plt.legend()
    plt.title("Prediction interval calibration"); plt.tight_layout()
    plt.savefig(plot_dir / "interval_calibration.png", dpi=180); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every metric distributed with Diffusion-TS")
    parser.add_argument("--output-dir", default="outputs/stock_condition_diffusion_may2026")
    parser.add_argument("--data-dir", default="train_data/scheme_a_glm_sequences")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    output_dir, data_dir = Path(args.output_dir), Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = np.load(output_dir / "forecast_samples_normalized.npy")
    real = np.load(output_dir / "truth_normalized.npy")
    history = np.load(output_dir / "history_normalized.npy")
    fake = samples[0]
    real_unit = np.clip((real + 1) / 2, 0, 1).astype(np.float32)
    fake_unit = np.clip((fake + 1) / 2, 0, 1).astype(np.float32)
    real_tensor, fake_tensor = torch.from_numpy(real_unit), torch.from_numpy(fake_unit)

    normalizer = RobustNormalizer.load(output_dir / "normalization.npz")
    real_original, fake_original = normalizer.inverse(real), normalizer.inverse(fake)
    mse = float(np.square(fake_original - real_original).mean())
    print(f"forecast MSE={mse:.8f}", flush=True)
    correlation = CrossCorrelLoss(real_tensor, name="CrossCorrelLoss").compute(fake_tensor).item()
    print(f"correlational score={correlation:.8f}", flush=True)

    discriminative, predictive, fids = [], [], []
    fake_device, real_device = fake_tensor, real_tensor
    for repeat in range(args.repeats):
        seed = 42 + repeat
        score, fake_accuracy, real_accuracy = discriminative_score(real_device, fake_device, device, seed)
        discriminative.append(score)
        print(f"discriminative {repeat + 1}/{args.repeats}: score={score:.6f} fake_acc={fake_accuracy:.6f} real_acc={real_accuracy:.6f}", flush=True)
    for repeat in range(args.repeats):
        score = predictive_score(real_device, fake_device, device, 142 + repeat)
        predictive.append(score)
        print(f"predictive {repeat + 1}/{args.repeats}: MAE={score:.6f}", flush=True)
    for repeat in range(args.repeats):
        score = context_fid(real_unit, fake_unit, device, 242 + repeat)
        fids.append(score)
        print(f"Context-FID {repeat + 1}/{args.repeats}: {score:.6f}", flush=True)

    feature_names = np.load(data_dir / "feature_names.npy", allow_pickle=False)[:19].tolist()
    save_plots(
        output_dir / "evaluation_plots", real, fake, samples, history,
        normalizer, feature_names,
    )
    results = {
        "forecast_mse_original_scale": mse,
        "correlational_score": correlation,
        "discriminative_score": summary(discriminative),
        "predictive_score": summary(predictive),
        "context_fid": summary(fids),
        "notes": {
            "metric_input": "robust-normalized values mapped from [-1,1] to [0,1]",
            "predictive_target": feature_names[-1],
            "generated_trajectory": "ensemble sample 0, matching one generated trajectory per real trajectory",
        },
    }
    destination = output_dir / "original_repo_metrics.json"
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved plots to {output_dir / 'evaluation_plots'}", flush=True)


if __name__ == "__main__":
    main()
