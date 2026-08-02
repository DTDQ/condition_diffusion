from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .model import ConditionalStockTransformer


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0, 0.999)


def extract(values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return values.gather(0, t).reshape(t.shape[0], *((1,) * (len(shape) - 1)))


class ConditionalDiffusionTS(nn.Module):
    def __init__(
        self,
        model: ConditionalStockTransformer,
        timesteps: int = 500,
        sampling_timesteps: int = 50,
        loss_type: str = "l1",
        fourier_weight: float | None = None,
        ddim_eta: float = 0.0,
        target_mode: str = "level",
        residual_scale: float = 2.0,
        condition_mode: str = "glm",
    ) -> None:
        super().__init__()
        if sampling_timesteps > timesteps:
            raise ValueError("sampling_timesteps cannot exceed timesteps")
        self.model = model
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.loss_type = loss_type
        self.fourier_weight = fourier_weight or math.sqrt(model.prediction_length) / 5
        self.ddim_eta = ddim_eta
        if target_mode not in {"level", "residual"}:
            raise ValueError("target_mode must be 'level' or 'residual'")
        if condition_mode not in {"glm", "zero"}:
            raise ValueError("condition_mode must be 'glm' or 'zero'")
        self.target_mode = target_mode
        self.residual_scale = residual_scale
        self.condition_mode = condition_mode

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        register = lambda name, value: self.register_buffer(name, value.float())
        register("betas", betas)
        register("alphas_cumprod", alpha_bar)
        register("sqrt_alphas_cumprod", torch.sqrt(alpha_bar))
        register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alpha_bar))
        register("sqrt_recip_alphas_cumprod", torch.sqrt(1 / alpha_bar))
        register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1 / alpha_bar - 1))
        register("loss_weight", torch.sqrt(alphas) * torch.sqrt(1 - alpha_bar) / betas / 100)

    def q_sample(self, clean: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_alphas_cumprod, t, clean.shape) * clean + extract(
            self.sqrt_one_minus_alphas_cumprod, t, clean.shape
        ) * noise

    def training_target(self, future: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        if self.target_mode == "residual":
            return (future - history[:, -1:, :]) / self.residual_scale
        return future

    def decode_sample(self, sample: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        if self.target_mode == "residual":
            return history[:, -1:, :] + self.residual_scale * sample
        return sample

    def prepare_condition(
        self, condition: torch.Tensor, condition_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.condition_mode == "zero":
            return torch.zeros_like(condition), torch.zeros_like(condition_mask)
        return condition, condition_mask

    def forward(
        self,
        future: torch.Tensor,
        history: torch.Tensor,
        condition: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = self.training_target(future, history)
        condition, condition_mask = self.prepare_condition(condition, condition_mask)
        batch = target.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch,), device=future.device)
        noisy = self.q_sample(target, t, torch.randn_like(target))
        prediction = self.model(noisy, t, history, condition, condition_mask)
        loss_fn = F.l1_loss if self.loss_type == "l1" else F.mse_loss
        time_per_sample = loss_fn(prediction, target, reduction="none").mean((1, 2))
        pred_fft = torch.fft.fft(prediction.transpose(1, 2), norm="forward")
        true_fft = torch.fft.fft(target.transpose(1, 2), norm="forward")
        freq_per_sample = (
            loss_fn(pred_fft.real, true_fft.real, reduction="none")
            + loss_fn(pred_fft.imag, true_fft.imag, reduction="none")
        ).mean((1, 2))
        weights = extract(self.loss_weight, t, target.shape).flatten()
        time_loss = (time_per_sample * weights).mean()
        freq_loss = (freq_per_sample * weights).mean()
        total = time_loss + self.fourier_weight * freq_loss
        return total, {"time_loss": time_loss.detach(), "frequency_loss": freq_loss.detach()}

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        condition: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        condition, condition_mask = self.prepare_condition(condition, condition_mask)
        batch, device = history.shape[0], history.device
        shape = (batch, self.model.prediction_length, self.model.target_dim)
        image = torch.randn(shape, device=device)
        times = torch.linspace(-1, self.num_timesteps - 1, self.sampling_timesteps + 1).int().tolist()
        pairs = list(zip(reversed(times[1:]), reversed(times[:-1])))
        for time, next_time in pairs:
            t = torch.full((batch,), time, device=device, dtype=torch.long)
            x0 = self.model(image, t, history, condition, condition_mask).clamp(-1, 1)
            noise = (extract(self.sqrt_recip_alphas_cumprod, t, image.shape) * image - x0) / extract(
                self.sqrt_recipm1_alphas_cumprod, t, image.shape
            )
            if next_time < 0:
                image = x0
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[next_time]
            sigma = self.ddim_eta * torch.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
            coefficient = torch.sqrt(torch.clamp(1 - alpha_next - sigma**2, min=0))
            image = x0 * alpha_next.sqrt() + coefficient * noise + sigma * torch.randn_like(image)
        return self.decode_sample(image, history).clamp(-1, 1)
