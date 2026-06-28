"""
Patch prediction head for A5 anatomy-prior experiments.

This module is intentionally small and independent so that both the A5
pretraining script and the A5 heatmap extraction script can import the same
head definition.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityInitializedPatchHead(nn.Module):
    """
    FC patch-prediction head: [B, N] -> [B, N].

    Default behavior is identity at initialization:
      - every Linear layer is initialized to identity
      - every bias is initialized to zero
      - relu_renorm keeps an already non-negative, sum-normalized attention
        map unchanged up to numerical epsilon.

    Recommended for A5:
      normalization="relu_renorm"

    Avoid normalization="softmax" if exact identity initialization matters:
    softmax(Ix) is not equal to x for a probability vector x.
    """

    def __init__(
        self,
        num_patches: int,
        num_layers: int = 2,
        normalization: str = "relu_renorm",
        eps: float = 1e-8,
    ):
        super().__init__()

        if int(num_patches) <= 0:
            raise ValueError(f"num_patches must be positive, got {num_patches}")
        if int(num_layers) <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.num_patches = int(num_patches)
        self.num_layers = int(num_layers)
        self.normalization = str(normalization).lower().strip()
        self.eps = float(eps)

        self.layers = nn.ModuleList(
            [nn.Linear(self.num_patches, self.num_patches) for _ in range(self.num_layers)]
        )
        self.reset_parameters_identity()

    @staticmethod
    def _init_linear_identity(layer: nn.Linear) -> None:
        if layer.in_features != layer.out_features:
            raise ValueError(
                "Identity initialization requires in_features == out_features, "
                f"got {layer.in_features} -> {layer.out_features}"
            )
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight.fill_diagonal_(1.0)
            if layer.bias is not None:
                layer.bias.zero_()

    def reset_parameters_identity(self) -> None:
        for layer in self.layers:
            self._init_linear_identity(layer)

    def forward(self, attn_patch_detached: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attn_patch_detached: Tensor [B, N]. It should already be detached
                by the caller. This method does not detach internally because
                the training script should explicitly log/check the detach.

        Returns:
            patch_pred: Tensor [B, N], normalized over N patches unless
                normalization="none".
        """
        if attn_patch_detached.ndim != 2:
            raise ValueError(
                f"Expected [B, N] input, got shape={tuple(attn_patch_detached.shape)}"
            )
        if attn_patch_detached.shape[-1] != self.num_patches:
            raise ValueError(
                f"Expected N={self.num_patches} patches, got N={attn_patch_detached.shape[-1]}"
            )

        x = attn_patch_detached
        for layer in self.layers:
            x = layer(x)

        if self.normalization == "none":
            return x

        if self.normalization == "relu_renorm":
            x = F.relu(x)
            return x / x.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        if self.normalization == "clamp_renorm":
            x = x.clamp_min(0.0)
            return x / x.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        if self.normalization == "softmax":
            return F.softmax(x, dim=-1)

        raise ValueError(
            f"Unknown normalization='{self.normalization}'. "
            "Use one of: relu_renorm, clamp_renorm, softmax, none."
        )


def infer_num_patches_from_config(config: dict) -> int:
    """
    Prefer explicit config['patch_head_num_patches']; otherwise infer 16x16
    patches from image_res/16, matching the 256 -> 16x16 setup.
    """
    if "patch_head_num_patches" in config:
        return int(config["patch_head_num_patches"])

    image_res = int(config.get("image_res", 256))
    patch_size = int(config.get("patch_size", 16))
    grid = image_res // patch_size
    return int(grid * grid)


def build_patch_head_from_config(config: dict) -> IdentityInitializedPatchHead:
    num_patches = infer_num_patches_from_config(config)
    num_layers = int(config.get("patch_head_num_layers", 2))
    normalization = str(config.get("patch_head_normalization", "relu_renorm"))
    eps = float(config.get("patch_head_eps", 1e-8))
    return IdentityInitializedPatchHead(
        num_patches=num_patches,
        num_layers=num_layers,
        normalization=normalization,
        eps=eps,
    )


def patch_vector_to_grid(patch_vec: torch.Tensor) -> torch.Tensor:
    """Convert [B, N] or [N] patch vector to [B, S, S]."""
    if patch_vec.ndim == 1:
        patch_vec = patch_vec.unsqueeze(0)
    if patch_vec.ndim != 2:
        raise ValueError(f"Expected [B, N] or [N], got {tuple(patch_vec.shape)}")

    n = int(patch_vec.shape[-1])
    side = int(math.sqrt(n))
    if side * side != n:
        raise ValueError(f"Number of patches must be square, got N={n}")
    return patch_vec.reshape(patch_vec.shape[0], side, side)


def upsample_patch_vector(
    patch_vec: torch.Tensor,
    target_size: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Upsample [B, N] patch map to [B, target_size, target_size]."""
    grid = patch_vector_to_grid(patch_vec).unsqueeze(1)  # [B,1,S,S]
    align_corners: Optional[bool] = False if mode in {"bilinear", "bicubic"} else None
    up = F.interpolate(
        grid.float(),
        size=(int(target_size), int(target_size)),
        mode=mode,
        align_corners=align_corners,
    )
    return up.squeeze(1)
