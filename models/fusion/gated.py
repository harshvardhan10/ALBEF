"""gated fusion, extracted unchanged from the thesis baseline."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ImageLevelGatedFusion(nn.Module):
    """Image-adaptive scalar weighting of original/lung/heart token streams.

    The gate reads only the three CLS tokens. It predicts one scalar weight per
    view for each study; softmax makes the weights non-negative and sum to one.
    The same three study-level weights are then applied to all 257 aligned ViT
    tokens. The final LayerNorm keeps the fused token scale well behaved.
    """

    def __init__(self, hidden_dim: int = 768, gate_hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, 3),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        # Exact equal weighting at initialization: softmax([0,0,0]) = 1/3.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def _validate(original_tokens, lung_tokens, heart_tokens):
        if not (original_tokens.shape == lung_tokens.shape == heart_tokens.shape):
            raise ValueError(
                "The three view token tensors must have identical shapes; got "
                f"original={tuple(original_tokens.shape)}, "
                f"lung={tuple(lung_tokens.shape)}, "
                f"heart={tuple(heart_tokens.shape)}"
            )
        if original_tokens.ndim != 3:
            raise ValueError(
                "Expected [B,N,D] view tokens; got "
                f"{tuple(original_tokens.shape)}"
            )

    def compute_weights(self, original_tokens, lung_tokens, heart_tokens):
        self._validate(original_tokens, lung_tokens, heart_tokens)
        if original_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dimension {self.hidden_dim}; "
                f"got {original_tokens.shape[-1]}"
            )
        cls_cat = torch.cat(
            [original_tokens[:, 0, :], lung_tokens[:, 0, :], heart_tokens[:, 0, :]],
            dim=-1,
        )
        return F.softmax(self.gate(cls_cat), dim=-1)

    def forward(
        self,
        original_tokens,
        lung_tokens,
        heart_tokens,
        return_weights: bool = False,
    ):
        weights = self.compute_weights(original_tokens, lung_tokens, heart_tokens)
        fused = (
            weights[:, 0, None, None] * original_tokens
            + weights[:, 1, None, None] * lung_tokens
            + weights[:, 2, None, None] * heart_tokens
        )
        fused = self.output_norm(fused)
        if return_weights:
            return fused, weights
        return fused
