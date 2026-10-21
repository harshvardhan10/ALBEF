"""mean fusion, extracted unchanged from the thesis baseline."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SimpleMeanFusion(nn.Module):
    """Parameter-free arithmetic mean of the three aligned token streams."""

    def forward(self, original_tokens, lung_tokens, heart_tokens):
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
        return (original_tokens + lung_tokens + heart_tokens) / 3.0
