from typing import List, Dict

import torch
import torch.nn.functional as F

# Globals to store cross-attention maps and grads
_cross_attn_maps: Dict[int, torch.Tensor] = {}
_cross_attn_grads: Dict[int, torch.Tensor] = {}


def _cross_attn_forward_hook(layer_idx: int):
    """
    Hook for multimodal cross-attention.
    Assumes output shape: (B, num_heads, T_text, N_img)
    """

    def hook(module, input, output):
        global _cross_attn_maps
        _cross_attn_maps[layer_idx] = output.detach()

    return hook


def _cross_attn_backward_hook(layer_idx: int):
    """
    Backward hook for cross-attention.
    grad_output[0]: (B, num_heads, T_text, N_img)
    """

    def hook(module, grad_input, grad_output):
        global _cross_attn_grads
        _cross_attn_grads[layer_idx] = grad_output[0].detach()

    return hook


def register_albef_crossattn_gradcam_hooks(model) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register hooks on ALL cross-attention layers in model.text_encoder.

    For ALBEF, cross-attn lives in:
      model.text_encoder.bert.encoder.layer[i].crossattention.self.dropout

    Hook the dropout *on the cross-attention* so that its forward output
    is the attention probabilities (after dropout) with shape (B, heads, T_text, N_img).
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []

    te = model.text_encoder  # BertForMaskedLM
    encoder = te.bert.encoder  # BertEncoder

    num_layers = len(encoder.layer)
    hooked_layers = []

    for layer_idx, layer in enumerate(encoder.layer):
        # Only some layers (6–11) have crossattention
        if hasattr(layer, "crossattention"):
            # crossattention is a BertAttention; its .self is BertSelfAttention
            sa = layer.crossattention.self
            dropout = sa.dropout  # this is applied to attention_probs

            h_fwd = dropout.register_forward_hook(_cross_attn_forward_hook(layer_idx))

            # Backward hook API differs slightly between PyTorch versions
            if hasattr(dropout, "register_full_backward_hook"):
                h_bwd = dropout.register_full_backward_hook(_cross_attn_backward_hook(layer_idx))
            else:
                h_bwd = dropout.register_backward_hook(_cross_attn_backward_hook(layer_idx))

            handles.extend([h_fwd, h_bwd])
            hooked_layers.append(layer_idx)

    print(
        f"[CrossAttn-GradCAM] Registered hooks on cross-attention dropout "
        f"for layers: {hooked_layers} (total {len(handles)} hooks)."
    )
    return handles



def remove_albef_crossattn_gradcam_hooks(handles: List[torch.utils.hooks.RemovableHandle]):
    for h in handles:
        h.remove()

    global _cross_attn_maps, _cross_attn_grads
    _cross_attn_maps.clear()
    _cross_attn_grads.clear()
    print("[CrossAttn-GradCAM] Hooks removed and buffers cleared.")


def _compute_crossattn_cam(
    num_img_tokens: int,
    device: torch.device,
    text_token_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Build image-token relevance from gradient-weighted cross-attention.

    _cross_attn_maps[layer_idx]:  (B, heads, T_text, N_img)
    _cross_attn_grads[layer_idx]: (B, heads, T_text, N_img)

    Strategy:
      - For each layer l:
          grad_attn = ReLU(A_l * G_l)
          aggregate over heads and text tokens -> (B, N_img)
      - Sum across layers.
      - Return (N_img,) for batch size 1.
    """
    global _cross_attn_maps, _cross_attn_grads

    layer_indices = sorted(_cross_attn_maps.keys())
    if not layer_indices:
        raise RuntimeError("No cross-attention maps captured. Did you call forward+backward?")

    any_attn = _cross_attn_maps[layer_indices[0]]
    B, H, T_text, N_img = any_attn.shape
    assert B == 1, "This helper assumes batch size 1."
    assert N_img == num_img_tokens, "Mismatch in number of image tokens."

    relevance = torch.zeros(N_img, device=device)

    for l in layer_indices:
        A = _cross_attn_maps[l].to(device)   # (1, heads, T_text, N_img)
        G = _cross_attn_grads[l].to(device)  # same shape

        grad_attn = A * G
        grad_attn = F.relu(grad_attn)        # (1, heads, T_text, N_img)

        # Optional: restrict to specific text tokens (e.g. label word)
        # text_token_mask: (T_text,) with 1.0 for tokens to keep
        if text_token_mask is not None:
            mask = text_token_mask.view(1, 1, T_text, 1)  # (1,1,T_text,1)
            grad_attn = grad_attn * mask

        # aggregate over heads and text tokens -> (1, N_img)
        grad_attn_mean = grad_attn.mean(dim=1).sum(dim=1)  # (1, N_img)

        relevance = relevance + grad_attn_mean[0]

    # normalize to [0,1]
    relevance = relevance - relevance.min()
    if relevance.max() > 0:
        relevance = relevance / (relevance.max() + 1e-6)
    else:
        relevance = torch.zeros_like(relevance)

    return relevance  # (N_img,)


def generate_albef_crossattn_gradcam(
    model,
    img_tensor: torch.Tensor,     # (1,3,H,W)
    input_ids: torch.Tensor,      # (1,T_text)
    attention_mask: torch.Tensor, # (1,T_text)
    device: torch.device,
    text_token_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Cross-attention-based Grad-CAM for one (image, label-text prompt).

    Steps:
      1. visual_encoder(image) -> image_embeds
      2. text_encoder in multimodal mode:
         - encoder_hidden_states=image_embeds
         - encoder_attention_mask=image_atts
      3. ITM logits via model.itm_head
      4. Backprop from 'match' logit to cross-attention maps
      5. Compute gradient-weighted cross-attention relevance over image tokens
      6. Reshape to (H', W')
    """
    global _cross_attn_maps, _cross_attn_grads
    _cross_attn_maps.clear()
    _cross_attn_grads.clear()

    model.eval()
    img_tensor = img_tensor.to(device, non_blocking=True)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # ---- Forward ----
    with torch.no_grad():
        image_embeds = model.visual_encoder(img_tensor)   # (1, N_img, C)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

    # Enable gradients for multimodal part
    image_embeds = image_embeds.detach()  # we don't need grads w.r.t. visual encoder
    image_embeds.requires_grad_(False)

    # Forward through text_encoder with encoder_hidden_states
    bert_out = model.text_encoder.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )
    sequence_output = bert_out.last_hidden_state  # (1, T, 768)
    multimodal_cls = sequence_output[:, 0, :]  # (1, 768)

    # ITM head: use [CLS] of multimodal text output
    itm_logits = model.itm_head(multimodal_cls)               # (1, 2) match / not-match
    match_logit = itm_logits[:, 1].squeeze(0)                 # scalar

    # ---- Backward ----
    model.zero_grad()
    match_logit.backward(retain_graph=False)

    # ---- Compute CAM ----
    B, N_img, _ = image_embeds.shape
    relevance = _compute_crossattn_cam(
        num_img_tokens=N_img,
        device=device,
        text_token_mask=text_token_mask,
    )  # (N_img,)

    # reshape to (H', W')
    num_patches = relevance.shape[0]
    grid = int(num_patches ** 0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Cannot reshape {num_patches} tokens into square grid.")

    cam = relevance.view(grid, grid)  # (H', W')
    return cam.cpu().float()
