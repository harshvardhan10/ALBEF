"""transformer fusion, extracted unchanged from the thesis baseline."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ViewTransformerFusion(nn.Module):
    """Fuse the three aligned view tokens at each ViT token position.

    Input tensors all have shape [B, N, D]. For each token position p, the
    original/lung/heart representations form a Transformer sequence of length 3.
    The three Transformer outputs are mean pooled, producing [B, N, D].

    Sequence-first layout [V, B*N, D] is used for compatibility with the older
    PyTorch version used by the ALBEF environment (no batch_first dependency).
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        num_layers = int(num_layers)
        mlp_ratio = float(mlp_ratio)
        dropout = float(dropout)

        if num_layers != 2:
            raise ValueError(
                "This ablation is defined as a 2-layer Transformer; "
                f"got num_layers={num_layers}"
            )
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.hidden_dim = hidden_dim
        self.num_views = 3

        # Learned identity of each input stream: original, lung, heart.
        # Shape is sequence-first so it broadcasts over B*N groups.
        self.view_embeddings = nn.Parameter(
            torch.zeros(self.num_views, 1, hidden_dim)
        )
        nn.init.normal_(self.view_embeddings, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(round(hidden_dim * mlp_ratio)),
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        original_tokens: torch.Tensor,
        lung_tokens: torch.Tensor,
        heart_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if not (
            original_tokens.shape == lung_tokens.shape == heart_tokens.shape
        ):
            raise ValueError(
                "The three view token tensors must have identical shapes; got "
                f"original={tuple(original_tokens.shape)}, "
                f"lung={tuple(lung_tokens.shape)}, "
                f"heart={tuple(heart_tokens.shape)}"
            )
        if original_tokens.ndim != 3:
            raise ValueError(
                "Expected view tokens with shape [B,N,D]; got "
                f"{tuple(original_tokens.shape)}"
            )

        batch_size, num_tokens, hidden_dim = original_tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dimension {self.hidden_dim}; got {hidden_dim}"
            )

        # [B,N,D] x 3 -> [3,B,N,D]
        view_sequence = torch.stack(
            [original_tokens, lung_tokens, heart_tokens],
            dim=0,
        )

        # Each spatial/token position becomes an independent length-3 sequence:
        # [3,B,N,D] -> [3,B*N,D]
        view_sequence = view_sequence.reshape(
            self.num_views,
            batch_size * num_tokens,
            hidden_dim,
        )
        view_sequence = view_sequence + self.view_embeddings

        # Self-attention occurs only across the 3 views for each token position.
        transformed = self.transformer(view_sequence)

        # Fixed symmetric aggregation after learned cross-view interaction.
        # [3,B*N,D] -> [B*N,D] -> [B,N,D]
        fused = transformed.mean(dim=0)
        fused = self.output_norm(fused)
        return fused.reshape(batch_size, num_tokens, hidden_dim)
