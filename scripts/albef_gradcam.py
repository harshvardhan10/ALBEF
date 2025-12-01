from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

# Global storage for hooks
_vit_feats = None  # forward activations
_vit_grads = None  # backward gradients


def _forward_hook(module, input, output):
    """
    Save forward activations from the target ViT block.
    output: (B, N+1, C)
    """
    global _vit_feats
    _vit_feats = output.detach()


def _backward_hook(module, grad_input, grad_output):
    """
    Save gradients w.r.t. the same output of the ViT block.
    grad_output[0]: (B, N+1, C)
    """
    global _vit_grads
    _vit_grads = grad_output[0].detach()


def _get_last_vit_block(model) -> torch.nn.Module:
    """
    Try to locate the last transformer block of the visual encoder.
    Works for ALBEF's VisionTransformer and timm-style ViTs.
    """
    ve = model.visual_encoder

    if hasattr(ve, "blocks"):  # common VisionTransformer pattern
        return ve.blocks[-1]
    if hasattr(ve, "encoder") and hasattr(ve.encoder, "layer"):
        # BERT-style encoder (less likely for ALBEF, but safe to support)
        return ve.encoder.layer[-1]

    raise AttributeError(
        "Could not find ViT blocks in model.visual_encoder. "
        "Expected attributes 'blocks' or 'encoder.layer'."
    )


def register_vit_gradcam_hooks(model) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register forward & backward hooks on ALBEF's last ViT block.
    Returns the list of hook handles (so you can remove them later).
    """
    target_block = _get_last_vit_block(model)

    # forward hook
    h_fwd = target_block.register_forward_hook(_forward_hook)

    # backward hook: use register_full_backward_hook if available
    if hasattr(target_block, "register_full_backward_hook"):
        h_bwd = target_block.register_full_backward_hook(_backward_hook)
    else:
        # older PyTorch fallback
        h_bwd = target_block.register_backward_hook(_backward_hook)

    return [h_fwd, h_bwd]


def remove_vit_gradcam_hooks(handles: List[torch.utils.hooks.RemovableHandle]):
    """
    Remove previously registered hooks.
    """
    for h in handles:
        h.remove()


def _compute_vit_gradcam_from_hooks(patch_grid: int = None) -> torch.Tensor:
    """
    Build Grad-CAM from stored ViT features & gradients.

    Returns:
        cam: torch.Tensor of shape (H', W') in [0,1]
    """
    global _vit_feats, _vit_grads

    if _vit_feats is None or _vit_grads is None:
        raise RuntimeError("Grad-CAM hooks have not captured any data yet.")

    # _vit_feats: (B, N+1, C)
    # _vit_grads: (B, N+1, C)
    feats = _vit_feats
    grads = _vit_grads

    if feats.dim() != 3:
        raise ValueError(f"Expected feats shape (B,N+1,C); got {feats.shape}")
    if grads.shape != feats.shape:
        raise ValueError(f"Mismatch feats {feats.shape} vs grads {grads.shape}")

    # Remove CLS token: keep only patch tokens
    feats = feats[:, 1:, :]  # (B, N, C)
    grads = grads[:, 1:, :]  # (B, N, C)

    # Assume batch=1 for Grad-CAM
    feats = feats.squeeze(0)  # (N, C)
    grads = grads.squeeze(0)  # (N, C)

    N, C = feats.shape

    # Determine patch grid automatically if not given
    grid_auto = int(np.sqrt(N))
    if grid_auto * grid_auto != N:
        raise ValueError(f"Cannot reshape N={N} tokens into square grid.")
    if patch_grid is None:
        patch_grid = grid_auto
    if patch_grid * patch_grid != N:
        raise ValueError(
            f"patch_grid={patch_grid} but N={N} (should be {patch_grid*patch_grid})"
        )

    # Channel-wise weights (Grad-CAM)
    # Mean gradients over spatial locations
    weights = grads.mean(dim=0)  # (C,)

    # Weighted sum over channels
    cam = (feats * weights.unsqueeze(0)).sum(dim=1)  # (N,)

    # Reshape to (H', W')
    cam = cam.view(patch_grid, patch_grid)  # (H', W')

    # ReLU + normalize
    cam = torch.relu(cam)
    if cam.max() > 0:
        cam = cam / (cam.max() + 1e-6)

    return cam  # (H', W') on CPU/GPU depending on feats


def generate_albef_gradcam(
    model,
    img_tensor: torch.Tensor,   # (1,3,H,W)
    text_feat: torch.Tensor,    # (D,) or (1,D), already normalized
    device: torch.device,
    patch_grid: int = None,
) -> torch.Tensor:
    """
    Run a full Grad-CAM pass for a single (image, label).

    Steps:
      - Forward image through visual encoder (with hooks attached).
      - Compute score = cosine(CLS, text_feat).
      - Backward from score.
      - Use stored ViT activations + gradients to compute CAM over patches.

    Returns:
      cam: torch.Tensor (H', W') in [0,1] on CPU.
    """
    global _vit_feats, _vit_grads
    _vit_feats = None
    _vit_grads = None

    model.eval()
    img_tensor = img_tensor.to(device, non_blocking=True)

    # We need gradients for internal features, but NOT necessarily for image pixels
    img_tensor.requires_grad_(False)

    text_feat = text_feat.to(device)
    if text_feat.dim() == 1:
        text_feat = text_feat.unsqueeze(0)  # (1,D)

    # ---- Forward pass through visual encoder (hooks will capture feats) ----
    image_embeds = model.visual_encoder(img_tensor)  # (1, N+1, 768)
    # CLS token at this level (before vision_proj)
    cls_tokens = image_embeds[:, 0, :]               # (1, 768)

    # Project CLS to joint space
    cls_proj = model.vision_proj(cls_tokens)         # (1, D)
    cls_proj = F.normalize(cls_proj, dim=-1)         # (1, D)

    # Ensure text_feat normalized too (if not already)
    text_feat = F.normalize(text_feat, dim=-1)       # (1, D)

    # ---- Scalar score for this label ----
    score = (cls_proj * text_feat).sum(dim=-1)       # (1,)
    score = score.squeeze(0)                         # scalar

    # ---- Backward: gradients w.r.t. last ViT block output ----
    model.zero_grad()
    score.backward(retain_graph=False)

    # ---- Build CAM from stored activations/gradients ----
    cam = _compute_vit_gradcam_from_hooks(patch_grid=patch_grid)  # (H', W')
    return cam.cpu()


def upsample_cam(cam: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Upsample a patch-level CAM to full image resolution.

    Args:
        cam: (H', W') tensor (on CPU or GPU)
        target_size: e.g. 256

    Returns:
        (target_size, target_size) float32 tensor on CPU in [0,1]
    """
    if not torch.is_tensor(cam):
        cam = torch.tensor(cam, dtype=torch.float32)

    cam_4d = cam.unsqueeze(0).unsqueeze(0)  # (1,1,H',W')
    cam_up = F.interpolate(
        cam_4d,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    cam_up = cam_up.squeeze(0).squeeze(0)
    cam_up = cam_up.clamp(0, 1)
    return cam_up.cpu().float()
