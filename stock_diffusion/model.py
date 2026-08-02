from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_UPSTREAM = Path(__file__).resolve().parents[1] / "third_party" / "Diffusion-TS"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

from Models.interpretable_diffusion.model_utils import Conv_MLP, LearnablePositionalEncoding  # noqa: E402
from Models.interpretable_diffusion.transformer import Decoder, Encoder  # noqa: E402


class GLMConditionEncoder(nn.Module):
    """Encode company/sector/macro/meta scores and their observed masks as 4 tokens."""

    GROUP_SIZES = (20, 12, 10, 6)

    def __init__(self, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(size * 2, model_dim),
                nn.SiLU(),
                nn.LayerNorm(model_dim),
                nn.Dropout(dropout),
                nn.Linear(model_dim, model_dim),
            )
            for size in self.GROUP_SIZES
        )
        self.type_embedding = nn.Parameter(torch.randn(4, model_dim) * 0.02)

    def forward(self, condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if condition.shape != mask.shape or condition.shape[-1] != 48:
            raise ValueError("condition and mask must both have shape [batch, 48]")
        tokens, offset = [], 0
        for encoder, width in zip(self.encoders, self.GROUP_SIZES):
            values = condition[:, offset : offset + width]
            observed = mask[:, offset : offset + width]
            tokens.append(encoder(torch.cat((values, observed), dim=-1)))
            offset += width
        return torch.stack(tokens, dim=1) + self.type_embedding.unsqueeze(0)


class ConditionalStockTransformer(nn.Module):
    """Diffusion-TS denoiser conditioned on history and four GLM tokens."""

    def __init__(
        self,
        target_dim: int = 19,
        history_length: int = 60,
        prediction_length: int = 20,
        model_dim: int = 128,
        encoder_layers: int = 2,
        decoder_layers: int = 3,
        heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.prediction_length = prediction_length
        self.future_embedding = Conv_MLP(target_dim, model_dim, resid_pdrop=dropout)
        self.history_embedding = Conv_MLP(target_dim, model_dim, resid_pdrop=dropout)
        self.inverse = Conv_MLP(model_dim, target_dim, resid_pdrop=dropout)
        self.future_position = LearnablePositionalEncoding(model_dim, dropout, prediction_length)
        self.history_position = LearnablePositionalEncoding(model_dim, dropout, history_length)
        self.noisy_encoder = Encoder(
            encoder_layers, model_dim, heads, dropout, dropout, mlp_ratio, "GELU"
        )
        history_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(history_layer, encoder_layers)
        self.condition_encoder = GLMConditionEncoder(model_dim, dropout)
        self.decoder = Decoder(
            prediction_length,
            target_dim,
            model_dim,
            heads,
            decoder_layers,
            dropout,
            dropout,
            mlp_ratio,
            "GELU",
            condition_dim=model_dim,
        )
        self.combine_season = nn.Conv1d(model_dim, target_dim, 1, bias=False)
        self.combine_mean = nn.Conv1d(decoder_layers, 1, 1, bias=False)

    def forward(
        self,
        noisy_future: torch.Tensor,
        timestep: torch.Tensor,
        history: torch.Tensor,
        condition: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_future.shape[1:] != (self.prediction_length, self.target_dim):
            raise ValueError("noisy_future has an unexpected shape")
        future_emb = self.future_position(self.future_embedding(noisy_future))
        noisy_memory = self.noisy_encoder(future_emb, timestep)
        history_memory = self.history_encoder(
            self.history_position(self.history_embedding(history))
        )
        glm_memory = self.condition_encoder(condition, condition_mask)
        memory = torch.cat((noisy_memory, history_memory, glm_memory), dim=1)

        output, means, trend, season = self.decoder(
            future_emb, timestep, memory, padding_masks=None
        )
        residual = self.inverse(output)
        residual_mean = residual.mean(dim=1, keepdim=True)
        season = self.combine_season(season.transpose(1, 2)).transpose(1, 2)
        season = season + residual - residual_mean
        trend = self.combine_mean(means) + residual_mean + trend
        return trend + season
