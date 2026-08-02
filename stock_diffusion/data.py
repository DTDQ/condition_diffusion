from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


TARGET_DIM = 19
CONDITION_DIM = 48
MASK_DIM = 48


@dataclass
class RobustNormalizer:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "RobustNormalizer":
        center = np.nanmedian(values, axis=(0, 1)).astype(np.float32)
        low = np.nanquantile(values, 0.01, axis=(0, 1)).astype(np.float32)
        high = np.nanquantile(values, 0.99, axis=(0, 1)).astype(np.float32)
        scale = np.maximum(np.maximum(center - low, high - center), 1e-6)
        return cls(center=center, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.clip((values - self.center) / self.scale, -1.0, 1.0).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.center

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, center=self.center, scale=self.scale)

    @classmethod
    def load(cls, path: Path) -> "RobustNormalizer":
        arrays = np.load(path)
        return cls(arrays["center"].astype(np.float32), arrays["scale"].astype(np.float32))


class StockWindowDataset(Dataset):
    """One sample is one stock's history and its following forecast target."""

    def __init__(
        self,
        data_dir: str | Path,
        split: Literal["train", "val", "test"],
        normalizer: RobustNormalizer,
        history_length: int = 60,
        prediction_length: int = 20,
        train_end: str = "2026-02-27",
        val_end: str = "2026-03-31",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.values = np.load(self.data_dir / "values.npy", mmap_mode="r")
        self.stock_ids = np.load(self.data_dir / "stock_ids.npy", allow_pickle=False)
        self.dates = np.load(self.data_dir / "dates.npy", allow_pickle=False).astype("datetime64[D]")
        self.normalizer = normalizer
        self.history_length = history_length
        self.prediction_length = prediction_length

        if self.values.ndim != 3 or self.values.shape[2] != TARGET_DIM + CONDITION_DIM + MASK_DIM:
            raise ValueError(f"Expected values [stock, date, 115], got {self.values.shape}")
        if self.values.shape[:2] != (len(self.stock_ids), len(self.dates)):
            raise ValueError("values, stock_ids and dates dimensions disagree")

        train_cut = np.datetime64(train_end)
        val_cut = np.datetime64(val_end)
        starts = np.arange(0, len(self.dates) - history_length - prediction_length + 1)
        future_start = self.dates[starts + history_length]
        future_end = self.dates[starts + history_length + prediction_length - 1]
        if split == "train":
            keep = future_end <= train_cut
        elif split == "val":
            keep = (future_start > train_cut) & (future_end <= val_cut)
        elif split == "test":
            keep = future_start > val_cut
        else:
            raise ValueError(f"Unknown split: {split}")
        self.starts = starts[keep]
        if len(self.starts) == 0:
            raise ValueError(f"No windows available for split={split}")

    def __len__(self) -> int:
        return len(self.stock_ids) * len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        stock_index, local_index = divmod(index, len(self.starts))
        start = int(self.starts[local_index])
        origin = start + self.history_length - 1
        future_start = origin + 1
        future_end = future_start + self.prediction_length

        history = self.normalizer.transform(np.asarray(
            self.values[stock_index, start : origin + 1, :TARGET_DIM]
        ))
        future = self.normalizer.transform(np.asarray(
            self.values[stock_index, future_start:future_end, :TARGET_DIM]
        ))
        condition = np.asarray(
            self.values[stock_index, origin, TARGET_DIM : TARGET_DIM + CONDITION_DIM]
        ).astype(np.float32)
        condition_mask = np.asarray(
            self.values[stock_index, origin, TARGET_DIM + CONDITION_DIM :]
        ).astype(np.float32)

        return {
            "history": torch.from_numpy(history),
            "future": torch.from_numpy(future),
            "condition": torch.from_numpy(condition),
            "condition_mask": torch.from_numpy(condition_mask),
            "stock_index": torch.tensor(stock_index),
            "origin_index": torch.tensor(origin),
        }


def fit_or_load_normalizer(
    data_dir: str | Path, output_path: str | Path, train_end: str
) -> RobustNormalizer:
    output_path = Path(output_path)
    if output_path.exists():
        return RobustNormalizer.load(output_path)
    data_dir = Path(data_dir)
    values = np.load(data_dir / "values.npy", mmap_mode="r")
    dates = np.load(data_dir / "dates.npy", allow_pickle=False).astype("datetime64[D]")
    train_values = np.asarray(values[:, dates <= np.datetime64(train_end), :TARGET_DIM])
    normalizer = RobustNormalizer.fit(train_values)
    normalizer.save(output_path)
    return normalizer
