"""
Attention-based (CheXzero-style) Grad-CAM for ALBEF's ViT.

Instead of using patch features from a single block, this module:
  - Hooks self-attention maps in ALL ViT blocks (visual_encoder.blocks[*].attn.attn_drop).
  - Captures gradients w.r.t. these attention maps.
  - Builds gradient-weighted attention per layer.
  - Applies attention rollout from CLS to patch tokens.
  - Returns a patch-level relevance map (H', W').

API is analogous to albef_gradcam.py:
  - register_albef_attn_gradcam_hooks(model)
  - remove_albef_attn_gradcam_hooks(handles)
  - generate_albef_attn_gradcam(model, img_tensor, text_feat, device)
  - upsample_cam(cam, target_size)
"""

from typing import List, Dict

import torch
import torch.nn.functional as F

# Globals for hooks (per-layer attention & gradients)
_attn_maps: Dict[int, torch.Tensor] = {}
_attn_grads: Dict[int, torch.Tensor] = {}


def _attn_forward_hook(layer_idx: int):
    """
    Forward hook for attention dropout.

    output shape: (B, num_heads, N, N)
      - N = number of tokens (CLS + patches)
    """

    def hook(module, input, output):
        global _attn_maps
        _attn_maps[layer_idx] = output.detach()

    return hook


def _attn_backward_hook(layer_idx: int):
    """
    Backward hook for attention dropout.

    grad_output[0] has shape: (B, num_heads, N, N),
    i.e. gradient w.r.t. the attention output.
    """

    def hook(module, grad_input, grad_output):
        global _attn_grads
        _attn_grads[layer_idx] = grad_output[0].detach()

    return hook


def register_albef_attn_gradcam_hooks(model) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register hooks on ALL attention layers in model.visual_encoder.

    We assume ALBEF.visual_encoder is a ViT with:
      - visual_encoder.blocks: ModuleList of Blocks
      - each Block has .attn.attn_drop module
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []

    ve = model.visual_encoder
    if not hasattr(ve, "blocks"):
        raise AttributeError(
            "visual_encoder has no 'blocks' attribute; "
            "attention Grad-CAM assumes ViT-style encoder."
        )

    for layer_idx, block in enumerate(ve.blocks):
        if not hasattr(block, "attn") or not hasattr(block.attn, "attn_drop"):
            raise AttributeError(
                f"Block {layer_idx} has no attn.attn_drop; cannot register attention hooks."
            )

        attn_drop = block.attn.attn_drop
        h_fwd = attn_drop.register_forward_hook(_attn_forward_hook(layer_idx))

        if hasattr(attn_drop, "register_full_backward_hook"):
            h_bwd = attn_drop.register_full_backward_hook(_attn_backward_hook(layer_idx))
        else:
            h_bwd = attn_drop.register_backward_hook(_attn_backward_hook(layer_idx))

        handles.extend([h_fwd, h_bwd])

    print(f"[Attn-GradCAM] Registered attention hooks on {len(ve.blocks)} blocks.")
    return handles


def remove_albef_attn_gradcam_hooks(handles: List[torch.utils.hooks.RemovableHandle]):
    """
    Remove all registered hooks and clear global buffers.
    """
    for h in handles:
        h.remove()

    global _attn_maps, _attn_grads
    _attn_maps.clear()
    _attn_grads.clear()

    print("[Attn-GradCAM] Hooks removed, buffers cleared.")


def _compute_attention_rollout_cam(
    num_tokens: int,
    device: torch.device,
    include_cls: bool = False,
) -> torch.Tensor:
    """
    Build patch-level relevance map using gradient-weighted attention rollout.

    Uses:
      _attn_maps[layer_idx]:  (B, heads, N, N)
      _attn_grads[layer_idx]: (B, heads, N, N)

    Steps:
      - For each layer l:
          A = attention map
          G = gradient w.r.t. A
          grad_attn = ReLU(A * G)       (gradient-weighted attention)
          grad_attn = mean over heads   -> (B, N, N)
          row-normalize
          add identity, renormalize
          rollout: R = A_hat @ R
      - After all layers:
          R[:, 0, :] = relevance of each token to CLS.
          Drop CLS and keep patch tokens only.

    Returns:
        relevance_patches: (N-1,) tensor for patch tokens.
    """
    global _attn_maps, _attn_grads

    layer_indices = sorted(_attn_maps.keys())
    if not layer_indices:
        raise RuntimeError("No attention maps captured. Did you run forward+backward?")

    # Initialize rollout R as identity
    # R: (B, N, N)
    any_attn = _attn_maps[layer_indices[0]]
    B = any_attn.shape[0]
    N = num_tokens
    R = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)

    for l in layer_indices:
        A = _attn_maps[l].to(device)   # (B, heads, N, N)
        G = _attn_grads[l].to(device)  # (B, heads, N, N)

        # Gradient-weighted attention
        grad_attn = A * G
        grad_attn = F.relu(grad_attn)

        # Average over heads -> (B, N, N)
        grad_attn = grad_attn.mean(dim=1)

        # Row-normalize
        grad_attn = grad_attn / (grad_attn.sum(dim=-1, keepdim=True) + 1e-6)

        # Add identity and renormalize (Chefer-style)
        I = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        A_hat = grad_attn + I
        A_hat = A_hat / (A_hat.sum(dim=-1, keepdim=True) + 1e-6)

        # Attention rollout
        R = torch.bmm(A_hat, R)   # (B, N, N)

    # Relevance from CLS token (index 0) to others
    cls_rel = R[:, 0, :]   # (B, N)

    if not include_cls:
        cls_rel = cls_rel[:, 1:]  # drop CLS, keep patch tokens only

    # We assume batch size 1 in this helper
    relevance = cls_rel[0]   # (N-1,)
    return relevance


def generate_albef_attn_gradcam(
    model,
    img_tensor: torch.Tensor,   # (1,3,H,W)
    text_feat: torch.Tensor,    # (D,) or (1,D)
    device: torch.device,
) -> torch.Tensor:
    """
    Attention-based Grad-CAM for one (image, label).

    1. Forward through visual_encoder to get tokens.
    2. Compute CLS feature via vision_proj, then cosine sim with text_feat.
    3. Backward from scalar score to ALL attention maps.
    4. Use gradient-weighted attention rollout to build CAM over patches.

    Returns:
        cam: (H', W') tensor in [0,1] on CPU.
    """
    global _attn_maps, _attn_grads
    _attn_maps.clear()
    _attn_grads.clear()

    model.eval()
    img_tensor = img_tensor.to(device, non_blocking=True)

    # Prepare text feature
    text_feat = text_feat.to(device)
    if text_feat.dim() == 1:
        text_feat = text_feat.unsqueeze(0)  # (1,D)
    text_feat = F.normalize(text_feat, dim=-1)  # (1,D)

    # ---- Forward: will trigger forward hooks on attention ----
    img_tensor.requires_grad_(False)
    image_tokens = model.visual_encoder(img_tensor)   # (1, N, 768); CLS + patches
    B, N, C = image_tokens.shape
    assert B == 1, "generate_albef_attn_gradcam assumes batch size 1."

    cls_tok = image_tokens[:, 0, :]                   # (1,768)
    cls_proj = model.vision_proj(cls_tok)             # (1,D)
    cls_proj = F.normalize(cls_proj, dim=-1)          # (1,D)

    score = (cls_proj * text_feat).sum(dim=-1)        # (1,)
    score = score.squeeze(0)                          # scalar

    # ---- Backward: triggers backward hooks on attention ----
    model.zero_grad()
    score.backward(retain_graph=False)

    # ---- Build CAM from attention maps + gradients ----
    rel_patches = _compute_attention_rollout_cam(
        num_tokens=N,
        device=device,
        include_cls=False,
    )  # (N-1,)

    # reshape to (H',W')
    num_patches = rel_patches.shape[0]
    grid_auto = int(num_patches ** 0.5)
    if grid_auto * grid_auto != num_patches:
        raise ValueError(f"Cannot reshape {num_patches} patches into a square grid.")
    cam = rel_patches.view(grid_auto, grid_auto)  # (H',W')

    # normalize to [0,1]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / (cam.max() + 1e-6)
    else:
        cam = torch.zeros_like(cam)

    return cam.cpu().float()


def upsample_cam(cam: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Upsample CAM to full image size.
    cam: (H', W') tensor
    Returns: (target_size, target_size) tensor in [0,1] on CPU.
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
