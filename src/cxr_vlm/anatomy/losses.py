import torch
import torch.nn.functional as F


def resize_prior_to_patch_mask(prior_mask, num_patches):
    """
    Resize cardiac prior mask to ViT patch grid.

    Args:
        prior_mask:
            Tensor [B, 1, H, W] or [B, H, W]

        num_patches:
            Number of patch tokens, e.g. 256 for image_res=256 and patch_size=16.

    Returns:
        prior_patch:
            Tensor [B, num_patches]
    """

    if prior_mask.dim() == 3:
        prior_mask = prior_mask.unsqueeze(1)

    grid = int(num_patches ** 0.5)

    if grid * grid != num_patches:
        raise ValueError(f"num_patches={num_patches} is not square.")

    prior_patch = F.interpolate(
        prior_mask.float(),
        size=(grid, grid),
        mode="area",
    )

    prior_patch = prior_patch.flatten(1)
    prior_patch = prior_patch.clamp(0.0, 1.0)

    return prior_patch


def support_outside_loss(attn_patch, prior_patch, active_mask=None):
    """
    Penalize attention mass outside cardiac anatomical support.

    Args:
        attn_patch:
            Tensor [B, N_patches].
            Normalized local patch attention.

        prior_patch:
            Tensor [B, N_patches].
            Resized cardiac prior support.

        active_mask:
            Optional Tensor [B].
            True where the anatomy support loss should be active.
    """

    if attn_patch.shape != prior_patch.shape:
        raise ValueError(
            f"Shape mismatch: attn_patch={attn_patch.shape}, prior_patch={prior_patch.shape}"
        )

    outside = 1.0 - prior_patch
    outside_mass = (attn_patch * outside).sum(dim=-1)

    if active_mask is not None:
        active_mask = active_mask.float()
        return (outside_mass * active_mask).sum() / active_mask.sum().clamp_min(1.0)

    return outside_mass.mean()