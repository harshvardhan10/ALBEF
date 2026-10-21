"""residual fusion, extracted unchanged from the thesis baseline."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class OriginalAnchoredResidualFusion(nn.Module):
    """Original-view anchor with small learned lung and heart residual gates.

    Two study-level sigmoid gates are predicted from the concatenated CLS
    representations. At initialization both gates equal `initial_gate` (0.1 by
    default), so the fused representation begins close to the original view:

        z_f = LayerNorm(z_original + g_lung*z_lung + g_heart*z_heart)
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        gate_hidden_dim: int = 768,
        initial_gate: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.initial_gate = float(initial_gate)
        if not 0.0 < self.initial_gate < 1.0:
            raise ValueError("initial_gate must be strictly between 0 and 1")

        self.gate = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, 2),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

        # Zero final weights make the initial gates input-independent. Choosing
        # bias=logit(initial_gate) yields exactly the requested small gate.
        nn.init.zeros_(self.gate[-1].weight)
        initial_logit = torch.log(
            torch.tensor(self.initial_gate / (1.0 - self.initial_gate))
        ).item()
        nn.init.constant_(self.gate[-1].bias, initial_logit)

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

    def compute_gates(self, original_tokens, lung_tokens, heart_tokens):
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
        return torch.sigmoid(self.gate(cls_cat))

    def forward(
        self,
        original_tokens,
        lung_tokens,
        heart_tokens,
        return_gates: bool = False,
    ):
        gates = self.compute_gates(original_tokens, lung_tokens, heart_tokens)
        fused = (
            original_tokens
            + gates[:, 0, None, None] * lung_tokens
            + gates[:, 1, None, None] * heart_tokens
        )
        fused = self.output_norm(fused)
        if return_gates:
            return fused, gates
        return fused
