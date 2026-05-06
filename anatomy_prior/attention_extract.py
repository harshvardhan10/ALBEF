from typing import List, Optional

import torch


def enable_crossattn_attention_saving_for_anatomy(model, layers: Optional[List[int]] = None):
    """
    Enable attention-map saving for ALBEF cross-attention layers.

    This is intentionally separate from scripts/albef_crossattn_gradcam.py
    to keep anatomy-prior training experiments isolated.
    """

    encoder = model.text_encoder.bert.encoder

    enabled = []

    for i, layer in enumerate(encoder.layer):
        if not hasattr(layer, "crossattention"):
            continue

        if layers is not None and i not in layers:
            continue

        sa = layer.crossattention.self

        if not hasattr(sa, "save_attention"):
            raise RuntimeError(
                f"Layer {i} crossattention.self does not support save_attention. "
                f"Check whether your xbert.py has attention saving enabled."
            )

        sa.save_attention = True
        enabled.append(i)

    print(f"[AnatomyPrior] Enabled cross-attention saving for layers: {enabled}")

    return enabled


def extract_raw_crossattn_for_anatomy_loss(
    model,
    text_token_mask: torch.Tensor,
    layers_to_use: Optional[List[int]] = None,
    remove_image_cls: bool = True,
    normalize_patches: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Extract differentiable raw text-to-image cross-attention.

    This function is for the anatomy-prior training loss.

    Args:
        model:
            ALBEF model.

        text_token_mask:
            Tensor [B, T_text].
            Selects pathology-relevant text tokens, e.g. Cardiomegaly.

        layers_to_use:
            Cross-attention layers to aggregate.
            Recommended first value: [8].

    Returns:
        attn_patch:
            Tensor [B, N_patches].
            Differentiable patch-level attention distribution.
    """

    encoder = model.text_encoder.bert.encoder

    available_layers = [
        i for i, layer in enumerate(encoder.layer)
        if hasattr(layer, "crossattention")
    ]

    if layers_to_use is None:
        use_layers = available_layers
    else:
        use_layers = [i for i in layers_to_use if i in available_layers]

    if len(use_layers) == 0:
        raise RuntimeError(
            f"No valid cross-attention layers selected. "
            f"Requested={layers_to_use}, available={available_layers}"
        )

    if text_token_mask.dim() != 2:
        raise ValueError(
            f"text_token_mask must have shape [B, T_text], got {text_token_mask.shape}"
        )

    text_token_mask = text_token_mask.float()

    collected = []

    for layer_idx in use_layers:
        sa = encoder.layer[layer_idx].crossattention.self

        if not hasattr(sa, "get_attention_map"):
            raise RuntimeError(
                f"Layer {layer_idx} does not expose get_attention_map(). "
                f"This requires your current ALBEF xbert.py attention-saving modification."
            )

        A = sa.get_attention_map()

        if A is None:
            raise RuntimeError(
                f"Layer {layer_idx} attention map is None. "
                f"Call enable_crossattn_attention_saving_for_anatomy() before the forward pass."
            )

        # A: [B, heads, T_text, N_img_tokens]
        B, H, T_text, N_img = A.shape

        if text_token_mask.shape[0] != B or text_token_mask.shape[1] != T_text:
            raise ValueError(
                f"text_token_mask shape {text_token_mask.shape} does not match "
                f"attention shape [B={B}, T_text={T_text}]"
            )

        if remove_image_cls:
            A = A[..., 1:]  # [B, heads, T_text, N_patches]

        qmask = text_token_mask[:, None, :, None]  # [B, 1, T_text, 1]

        token_denom = qmask.sum(dim=2).clamp_min(eps)
        A = (A * qmask).sum(dim=2) / token_denom  # [B, heads, N_patches]

        A = A.mean(dim=1)  # [B, N_patches]

        collected.append(A)

    attn_patch = torch.stack(collected, dim=0).mean(dim=0)

    if normalize_patches:
        attn_patch = attn_patch / attn_patch.sum(dim=-1, keepdim=True).clamp_min(eps)

    return attn_patch