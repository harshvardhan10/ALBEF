from typing import List
import numpy as np
import torch
import torch.nn.functional as F

# Globals for hooks
_vit_feats_in = None   # block INPUT activations
_vit_grads_in = None   # block INPUT gradients


def _forward_hook(module, input, output):
    """
    Save INPUT activations to the last ViT block.
    input: tuple, input[0] has shape (B, N+1, C)
    """
    global _vit_feats_in
    _vit_feats_in = input[0].detach()


def _backward_hook(module, grad_input, grad_output):
    """
    Save gradients w.r.t. the INPUT of the last ViT block.
    grad_input[0]: (B, N+1, C)
    """
    global _vit_grads_in
    _vit_grads_in = grad_input[0].detach()


def _get_last_vit_block(model) -> torch.nn.Module:
    """
    Pick a ViT block from ALBEF.visual_encoder for Grad-CAM.
    For ALBEF, visual_encoder.blocks is a ModuleList of 12 Block's (0..11).
    Choosing blocks[-3] (i.e. index 9) to be slightly earlier than the last block,
    which tends to give better spatial localization.
    """
    ve = model.visual_encoder

    if hasattr(ve, "blocks"):
        n_blocks = len(ve.blocks)  # should be 12
        if n_blocks >= 3:
            idx = n_blocks - 3     # 10th block     
        else:
            idx = n_blocks - 1     # fallback to last block

        print(f"[Grad-CAM] Using visual_encoder.blocks[{idx}] for CAM")
        return ve.blocks[idx]

    raise AttributeError(
        "visual_encoder has no 'blocks' attribute; this Grad-CAM helper "
        "assumes a ViT-like visual encoder with .blocks."
    )


def register_vit_gradcam_hooks(model) -> List[torch.utils.hooks.RemovableHandle]:
    block = _get_last_vit_block(model)
    h_fwd = block.register_forward_hook(_forward_hook)
    # full backward hook if available
    if hasattr(block, "register_full_backward_hook"):
        h_bwd = block.register_full_backward_hook(_backward_hook)
    else:
        h_bwd = block.register_backward_hook(_backward_hook)
    return [h_fwd, h_bwd]


def remove_vit_gradcam_hooks(handles: List[torch.utils.hooks.RemovableHandle]):
    for h in handles:
        h.remove()


def _compute_vit_gradcam_from_hooks(patch_grid: int = None) -> torch.Tensor:
    """
    Use INPUT features & gradients from the last ViT block to build CAM.

    Returns:
        cam: (H', W') tensor in [0,1] (after normalization, no ReLU by default here)
    """
    global _vit_feats_in, _vit_grads_in

    if _vit_feats_in is None or _vit_grads_in is None:
        raise RuntimeError("Grad-CAM hooks didn't capture data. Did you run forward+backward?")

    feats = _vit_feats_in.clone()   # (B, N+1, C)
    grads = _vit_grads_in.clone()   # (B, N+1, C)

    # remove CLS (token 0)
    feats = feats[:, 1:, :]   # (B, N, C)
    grads = grads[:, 1:, :]   # (B, N, C)

    feats = feats.squeeze(0)  # (N, C)
    grads = grads.squeeze(0)  # (N, C)

    N, C = feats.shape

    # infer patch grid
    grid_auto = int(np.sqrt(N))
    if grid_auto * grid_auto != N:
        raise ValueError(f"Cannot reshape N={N} tokens into a square grid.")
    if patch_grid is None:
        patch_grid = grid_auto
    if patch_grid * patch_grid != N:
        raise ValueError(
            f"patch_grid={patch_grid}, but N={N} (expected {patch_grid*patch_grid})."
        )

    # Grad-CAM weights: mean over spatial positions
    weights = grads.mean(dim=0)   # (C,)

    # Weighted combination of features
    cam = (feats * weights.unsqueeze(0)).sum(dim=1)  # (N,)

    # reshape
    cam = cam.view(patch_grid, patch_grid)  # (H', W')

    # normalize to [0,1] (WITHOUT ReLU first)
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
    else:
        cam = torch.zeros_like(cam)

    return cam  # (H', W')


def generate_albef_gradcam(
    model,
    img_tensor: torch.Tensor,   # (1,3,H,W)
    text_feat: torch.Tensor,    # (D,) or (1,D)
    device: torch.device,
    patch_grid: int = None,
) -> torch.Tensor:
    """
    Full Grad-CAM for one (image, label).

    1. Forward through visual_encoder (hooks see block input).
    2. Compute score = cosine(CLS(image), text_feat).
    3. Backward from score.
    4. Use stored INPUT features & grads to build CAM.
    """
    global _vit_feats_in, _vit_grads_in
    _vit_feats_in = None
    _vit_grads_in = None

    model.eval()
    img_tensor = img_tensor.to(device, non_blocking=True)

    text_feat = text_feat.to(device)
    if text_feat.dim() == 1:
        text_feat = text_feat.unsqueeze(0)  # (1,D)
    text_feat = F.normalize(text_feat, dim=-1)

    # ---- Forward: triggers forward hook (captures block input) ----
    img_tensor.requires_grad_(False)
    image_embeds = model.visual_encoder(img_tensor)      # (1, N+1, 768)

    cls_tokens = image_embeds[:, 0, :]                   # (1,768)
    cls_proj = model.vision_proj(cls_tokens)             # (1,D)
    cls_proj = F.normalize(cls_proj, dim=-1)             # (1,D)

    score = (cls_proj * text_feat).sum(dim=-1)           # (1,)
    score = score.squeeze(0)                             # scalar

    # ---- Backward: triggers backward hook (captures grad_input) ----
    model.zero_grad()
    score.backward(retain_graph=False)

    # ---- Build CAM from stored features+grads ----
    cam = _compute_vit_gradcam_from_hooks(patch_grid=patch_grid)  # (H', W')
    return cam.cpu()


def upsample_cam(cam: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Upsample CAM to full image size.
    cam: (H', W') tensor
    """
    if not torch.is_tensor(cam):
        cam = torch.tensor(cam, dtype=torch.float32)

    cam_4d = cam.unsqueeze(0).unsqueeze(0)          # (1,1,H',W')
    cam_up = F.interpolate(
        cam_4d,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    cam_up = cam_up.squeeze(0).squeeze(0)
    cam_up = cam_up.clamp(0, 1)
    return cam_up.cpu().float()
