from typing import List, Dict, Optional

import torch
import torch.nn.functional as F

# Globals to store cross-attention maps and grads (fallback path)
_cross_attn_maps: Dict[int, torch.Tensor] = {}
_cross_attn_grads: Dict[int, torch.Tensor] = {}


def _cross_attn_forward_hook(layer_idx: int):
    """
    Hook for multimodal cross-attention dropout output (attention_probs after dropout).
    output shape: (B, heads, T_text, N_img)
    """
    def hook(module, input, output):
        global _cross_attn_maps
        _cross_attn_maps[layer_idx] = output.detach()
    return hook


def _cross_attn_backward_hook(layer_idx: int):
    """
    Backward hook for cross-attention dropout.
    grad_output[0]: (B, heads, T_text, N_img)
    """
    def hook(module, grad_input, grad_output):
        global _cross_attn_grads
        _cross_attn_grads[layer_idx] = grad_output[0].detach()
    return hook


def register_albef_crossattn_gradcam_hooks(model) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register hooks on ALL cross-attention layers in model.text_encoder (fallback mechanism).

    For ALBEF, cross-attn lives in:
      model.text_encoder.bert.encoder.layer[i].crossattention.self.dropout

    Hook the dropout on the cross-attention so that its forward output is attention_probs
    with shape (B, heads, T_text, N_img).
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []

    te = model.text_encoder
    encoder = te.bert.encoder

    hooked_layers = []
    for layer_idx, layer in enumerate(encoder.layer):
        if hasattr(layer, "crossattention"):
            sa = layer.crossattention.self  # BertSelfAttention
            dropout = sa.dropout

            h_fwd = dropout.register_forward_hook(_cross_attn_forward_hook(layer_idx))
            if hasattr(dropout, "register_full_backward_hook"):
                h_bwd = dropout.register_full_backward_hook(_cross_attn_backward_hook(layer_idx))
            else:
                h_bwd = dropout.register_backward_hook(_cross_attn_backward_hook(layer_idx))

            handles.extend([h_fwd, h_bwd])
            hooked_layers.append(layer_idx)

    print(
        f"[CrossAttn-GradCAM] Registered fallback hooks on cross-attention dropout "
        f"for layers: {hooked_layers}."
    )
    return handles


def remove_albef_crossattn_gradcam_hooks(handles: List[torch.utils.hooks.RemovableHandle]):
    for h in handles:
        h.remove()

    global _cross_attn_maps, _cross_attn_grads
    _cross_attn_maps.clear()
    _cross_attn_grads.clear()
    print("[CrossAttn-GradCAM] Hooks removed and buffers cleared.")


# ---------------------------
# Normalization helpers
# ---------------------------

def normalize_cam_vis_from_raw(patch_relevance_raw: torch.Tensor) -> torch.Tensor:
    """
    Visualization normalization:
      - ReLU
      - divide by max
    No min-subtraction.
    """
    x = F.relu(patch_relevance_raw)
    mx = x.max()
    if mx > 0:
        x = x / (mx + 1e-6)
    else:
        x = torch.zeros_like(x)
    return x


# ---------------------------
# CAM computation
# ---------------------------

def _compute_crossattn_relevance_tokens(
    num_img_tokens: int,
    device: torch.device,
    text_token_mask: Optional[torch.Tensor] = None,
    layers_to_use: Optional[List[int]] = None,
    prefer_getters: bool = True,
    model=None,
) -> torch.Tensor:
    """
    Return relevance over ALL image tokens (CLS + patches): shape (N_img,).

    Two modes:
      A) Preferred: read attention map + grad via get_attention_map()/get_attn_gradients()
         from each crossattention.self module, if available and prefer_getters=True.
      B) Fallback: use global buffers _cross_attn_maps/_cross_attn_grads collected via hooks.
    """
    # -------- Mode A: module getters --------
    if prefer_getters and model is not None:
        te = model.text_encoder
        encoder = te.bert.encoder

        # identify available crossattention layers
        avail = [i for i, layer in enumerate(encoder.layer) if hasattr(layer, "crossattention")]
        if layers_to_use is None:
            use_layers = avail
        else:
            use_layers = [l for l in layers_to_use if l in avail]
            if len(use_layers) == 0:
                raise RuntimeError(f"Requested layers {layers_to_use} not available. Cross-attn layers: {avail}")

        # Check whether getters exist (on at least one layer)
        any_has_getters = False
        for l in use_layers:
            sa = encoder.layer[l].crossattention.self
            if hasattr(sa, "get_attention_map") and hasattr(sa, "get_attn_gradients"):
                any_has_getters = True
                break

        if any_has_getters:
            relevance = None

            for l in use_layers:
                sa = encoder.layer[l].crossattention.self
                if not (hasattr(sa, "get_attention_map") and hasattr(sa, "get_attn_gradients")):
                    raise RuntimeError(
                        f"Layer {l} crossattention.self does not expose get_attention_map/get_attn_gradients "
                        f"but another layer did. Please make this consistent or set prefer_getters=False."
                    )

                A = sa.get_attention_map()      # (B, heads, T_text, N_img)
                G = sa.get_attn_gradients()     # (B, heads, T_text, N_img)

                if A is None or G is None:
                    raise RuntimeError(
                        f"Layer {l}: attention map / gradients are None. Ensure your attention module stores them "
                        f"during forward/backward."
                    )

                A = A.to(device)
                G = G.to(device)

                B, H, T_text, N_img = A.shape
                assert B == 1, "Batch size 1 expected."
                assert N_img == num_img_tokens, f"Mismatch N_img={N_img} vs expected {num_img_tokens}"

                grad_attn = F.relu(A * G)

                if text_token_mask is not None:
                    # accept either (T,) or (1,T)
                    if text_token_mask.dim() == 2:
                        tmask = text_token_mask[0]
                    else:
                        tmask = text_token_mask
                    tmask = tmask.to(device)
                    assert tmask.numel() == T_text, f"text_token_mask len {tmask.numel()} != T_text {T_text}"
                    grad_attn = grad_attn * tmask.view(1, 1, T_text, 1)

                layer_rel = grad_attn.mean(dim=1).sum(dim=1)[0]  # (N_img,)

                relevance = layer_rel if relevance is None else (relevance + layer_rel)

            return relevance  # (N_img,)

    # -------- Mode B: fallback buffers from hooks --------
    global _cross_attn_maps, _cross_attn_grads

    all_layers = sorted(_cross_attn_maps.keys())
    if not all_layers:
        raise RuntimeError(
            "No cross-attention maps captured in fallback buffers. "
            "Did you register hooks and run forward+backward?"
        )

    if layers_to_use is None:
        layer_indices = all_layers
    else:
        layer_indices = [l for l in layers_to_use if l in _cross_attn_maps]
        if not layer_indices:
            raise RuntimeError(
                f"Requested layers {layers_to_use} not found in cross-attn maps. Available: {all_layers}"
            )

    any_attn = _cross_attn_maps[layer_indices[0]]
    B, H, T_text, N_img = any_attn.shape
    assert B == 1, "Batch size 1 expected."
    assert N_img == num_img_tokens, f"Mismatch N_img={N_img} vs expected {num_img_tokens}"

    relevance = torch.zeros(N_img, device=device)

    for l in layer_indices:
        A = _cross_attn_maps[l].to(device)
        G = _cross_attn_grads[l].to(device)

        grad_attn = F.relu(A * G)

        if text_token_mask is not None:
            if text_token_mask.dim() == 2:
                tmask = text_token_mask[0]
            else:
                tmask = text_token_mask
            tmask = tmask.to(device)
            grad_attn = grad_attn * tmask.view(1, 1, T_text, 1)

        layer_rel = grad_attn.mean(dim=1).sum(dim=1)[0]  # (N_img,)
        relevance = relevance + layer_rel

    return relevance  # (N_img,)


def generate_albef_crossattn_gradcam(
    model,
    img_tensor: torch.Tensor,     # (1,3,H,W)
    input_ids: torch.Tensor,      # (1,T_text)
    attention_mask: torch.Tensor, # (1,T_text)
    device: torch.device,
    text_token_mask: Optional[torch.Tensor] = None,
    layers_to_use: Optional[List[int]] = None,
    prefer_getters: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Cross-attention Grad-CAM for one (image, label-text).

    Uses ITM match logit as objective (ALBEF_itm).
    Returns:
      {
        "cam_raw": (H', W') float32 tensor (ReLU applied at the end is NOT forced),
        "cam_vis": (H', W') float32 tensor in [0,1] for visualization/thresholding
      }

    Notes:
      - relevance is computed over (CLS + patches), then CLS is dropped.
      - cam_raw is the patch relevance BEFORE visualization scaling (but still aggregated across layers).
      - cam_vis is ReLU(raw) / max.
    """
    model.eval()

    # Clear fallback buffers (safe even if using getters)
    global _cross_attn_maps, _cross_attn_grads
    _cross_attn_maps.clear()
    _cross_attn_grads.clear()

    img_tensor = img_tensor.to(device, non_blocking=True)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # ---- visual encoder ----
    with torch.no_grad():
        image_embeds = model.visual_encoder(img_tensor)  # (1, 257, 768)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

    image_embeds = image_embeds.detach()

    # ---- multimodal forward ----
    bert_out = model.text_encoder.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )
    sequence_output = bert_out.last_hidden_state     # (1, T, 768)
    multimodal_cls = sequence_output[:, 0, :]        # (1, 768)

    itm_logits = model.itm_head(multimodal_cls)      # (1, 2)
    match_logit = itm_logits[:, 1].squeeze(0)        # scalar

    # ---- backward ----
    model.zero_grad(set_to_none=True)
    match_logit.backward(retain_graph=False)

    # ---- relevance over image tokens ----
    relevance_tokens = _compute_crossattn_relevance_tokens(
        num_img_tokens=image_embeds.shape[1],
        device=device,
        text_token_mask=text_token_mask,
        layers_to_use=layers_to_use,
        prefer_getters=prefer_getters,
        model=model,
    )  # (257,)

    # Drop CLS
    patch_relevance_raw = relevance_tokens[1:]  # (256,)

    # Reshape
    num_patches = patch_relevance_raw.numel()
    grid = int(num_patches ** 0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Cannot reshape {num_patches} patch tokens into square grid.")

    cam_raw = patch_relevance_raw.view(grid, grid)

    # cam_vis: your requested normalization
    cam_vis = normalize_cam_vis_from_raw(patch_relevance_raw).view(grid, grid)

    return {
        "cam_raw": cam_raw.detach().cpu().float(),
        "cam_vis": cam_vis.detach().cpu().float(),
    }


def enable_crossattn_attention_saving(model, layers=None):
    """
    Turn on saving attention maps + gradients for cross-attention self-attn modules.
    layers: list of layer indices to enable (e.g. [8,9,10,11]); if None, enable all cross-attn layers.
    """
    encoder = model.text_encoder.bert.encoder
    enabled = []
    for i, layer in enumerate(encoder.layer):
        if not hasattr(layer, "crossattention"):
            continue
        if layers is not None and i not in layers:
            continue
        sa = layer.crossattention.self  # BertSelfAttention
        sa.save_attention = True
        enabled.append(i)
    print(f"[CrossAttn-GradCAM] Enabled save_attention for layers: {enabled}")
    return enabled
