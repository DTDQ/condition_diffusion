import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stock_diffusion.data import RobustNormalizer, StockWindowDataset
from stock_diffusion.diffusion import ConditionalDiffusionTS
from stock_diffusion.model import ConditionalStockTransformer, GLMConditionEncoder


class StockDiffusionTest(unittest.TestCase):
    def test_condition_mask_changes_token(self):
        encoder = GLMConditionEncoder(32, 0.0).eval()
        values = torch.zeros(2, 48)
        masks = torch.stack((torch.zeros(48), torch.ones(48)))
        output = encoder(values, masks)
        self.assertEqual(output.shape, (2, 4, 32))
        self.assertFalse(torch.allclose(output[0], output[1]))

    def test_model_loss_and_sampling_shapes(self):
        model = ConditionalStockTransformer(
            history_length=6, prediction_length=4, model_dim=32,
            encoder_layers=1, decoder_layers=1, heads=4, dropout=0.0,
        )
        diffusion = ConditionalDiffusionTS(model, timesteps=8, sampling_timesteps=2)
        history = torch.randn(2, 6, 19)
        future = torch.randn(2, 4, 19).clamp(-1, 1)
        condition = torch.randn(2, 48)
        mask = torch.ones(2, 48)
        loss, _ = diffusion(future, history, condition, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        sample = diffusion.sample(history, condition, mask)
        self.assertEqual(sample.shape, future.shape)

    def test_residual_target_round_trip_and_zero_condition(self):
        model = ConditionalStockTransformer(
            history_length=6, prediction_length=4, model_dim=32,
            encoder_layers=1, decoder_layers=1, heads=4, dropout=0.0,
        )
        diffusion = ConditionalDiffusionTS(
            model, timesteps=8, sampling_timesteps=2,
            target_mode="residual", residual_scale=2.0, condition_mode="zero",
        )
        history = torch.randn(2, 6, 19).clamp(-1, 1)
        future = torch.randn(2, 4, 19).clamp(-1, 1)
        target = diffusion.training_target(future, history)
        self.assertTrue(torch.allclose(diffusion.decode_sample(target, history), future))
        condition, mask = diffusion.prepare_condition(torch.ones(2, 48), torch.ones(2, 48))
        self.assertEqual(condition.count_nonzero().item(), 0)
        self.assertEqual(mask.count_nonzero().item(), 0)

    def test_dataset_window_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "values.npy", np.random.randn(2, 12, 115).astype(np.float32))
            np.save(root / "stock_ids.npy", np.array(["a", "b"]))
            dates = np.arange("2026-01-01", "2026-01-13", dtype="datetime64[D]")
            np.save(root / "dates.npy", dates.astype(str))
            normalizer = RobustNormalizer(np.zeros(19, np.float32), np.ones(19, np.float32))
            dataset = StockWindowDataset(
                root, "train", normalizer, history_length=4, prediction_length=2,
                train_end="2026-01-12", val_end="2026-01-12",
            )
            item = dataset[0]
            self.assertEqual(item["history"].shape, (4, 19))
            self.assertEqual(item["future"].shape, (2, 19))
            self.assertEqual(item["condition"].shape, (48,))
            self.assertEqual(item["condition_mask"].shape, (48,))


if __name__ == "__main__":
    unittest.main()
