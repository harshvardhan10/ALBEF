"""FC projection with legacy top-level weight/bias keys.

Normalization stays on ALBEF.view_fusion_norm for checkpoint compatibility.
"""
import torch
from torch import nn


class FullyConnectedFusion(nn.Linear):
    def __init__(self, hidden_dim):
        super().__init__(3 * hidden_dim, hidden_dim)
        with torch.no_grad():
            self.weight.zero_()
            self.bias.zero_()
            eye = torch.eye(hidden_dim, dtype=self.weight.dtype)
            for index in range(3):
                self.weight[:, index*hidden_dim:(index+1)*hidden_dim].copy_(eye / 3.)

    def forward(self, original_tokens, lung_tokens, heart_tokens):
        if not (original_tokens.shape == lung_tokens.shape == heart_tokens.shape):
            raise ValueError("The three view token tensors must have identical shapes")
        return super().forward(torch.cat([original_tokens, lung_tokens, heart_tokens], dim=-1))
