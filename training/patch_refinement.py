"""Patch-refinement helpers with intentionally distinct A5/A6.1 reductions.

A5 clamps the weighted denominator at 1; A6.1 uses eps. Do not silently
merge these definitions when reviewing the trainable/frozen workflows.
"""
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

def get_model_without_ddp(model):
    return model.module if hasattr(model, "module") else model


def weighted_mean_a5(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights.clamp_min(0.0)
    return (values * w).sum() / w.sum().clamp_min(1.0)


def module_grad_norm_a5(module: nn.Module) -> float:
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total_sq += float(g.norm(2).item() ** 2)
    return float(total_sq ** 0.5)


def module_param_norm_a5(module: nn.Module) -> float:
    total_sq = 0.0
    for p in module.parameters():
        with torch.no_grad():
            total_sq += float(p.detach().float().norm(2).item() ** 2)
    return float(total_sq ** 0.5)


def sync_module_gradients_a5(module: nn.Module) -> None:
    """
    Manually average gradients for a module that is intentionally kept outside DDP.

    A5 uses patch_head outside ALBEF.forward(), so patch_head must not be registered
    inside the DDP-wrapped ALBEF module. This function keeps patch_head synchronized
    across ranks by all-reducing its gradients after loss.backward().
    """
    if not (dist.is_available() and dist.is_initialized()):
        return

    world_size = float(dist.get_world_size())
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def load_model_albef_state_a5(model, state_dict, strict=False):
    """
    Load ALBEF weights into a model that may already contain patch_head.
    strict=False is intentional because old A0-A3 checkpoints do not have
    patch_head.* keys, and A5 checkpoint['model'] deliberately excludes them.
    """
    msg = model.load_state_dict(state_dict, strict=strict)
    if hasattr(msg, "missing_keys") or hasattr(msg, "unexpected_keys"):
        missing = list(getattr(msg, "missing_keys", []))
        unexpected = list(getattr(msg, "unexpected_keys", []))
        allowed_missing = [k for k in missing if k.startswith("patch_head.")]
        other_missing = [k for k in missing if not k.startswith("patch_head.")]
        if other_missing:
            print(f"[Checkpoint] Non-patch missing keys: {other_missing[:20]} ...")
        if unexpected:
            print(f"[Checkpoint] Unexpected keys: {unexpected[:20]} ...")
        if allowed_missing:
            print(f"[Checkpoint] Missing patch_head keys are expected: {allowed_missing[:4]} ...")
    return msg


def weighted_mean_a6(values, weights, eps=1e-8):
    weights = weights.float().clamp_min(0.0)
    return (values * weights).sum() / weights.sum().clamp_min(eps)


def module_grad_norm_a6(module):
    total = 0.0
    found = False
    for p in module.parameters():
        if p.grad is not None:
            found = True
            total += float(p.grad.detach().float().norm(2).cpu()) ** 2
    return total ** 0.5 if found else 0.0


def module_param_norm_a6(module):
    total = 0.0
    for p in module.parameters():
        total += float(p.detach().float().norm(2).cpu()) ** 2
    return total ** 0.5


def sync_module_gradients_a6(module):
    """
    Average standalone patch_head gradients across ranks.

    Also handles ranks that had no active support samples by filling None grads
    with zeros before all_reduce. This keeps all ranks synchronized.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return

    world_size = dist.get_world_size()
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def load_model_albef_state_a6(model, state_dict, strict=False):
    msg = model.load_state_dict(state_dict, strict=strict)
    print(f"[Checkpoint:A6.1] load_state_dict strict={strict}: {msg}")


def get_patch_head(model):
    raw_model = get_model_without_ddp(model)
    if not hasattr(raw_model, "patch_head"):
        raise AttributeError("A5 requires raw_model.patch_head, but it is missing.")
    return raw_model.patch_head


def albef_state_dict_without_patch_head(model_without_ddp):
    """
    Keep checkpoint['model'] loadable by old zero-shot/evaluation code that
    instantiates plain ALBEF. Patch head is saved separately.
    """
    return {
        k: v for k, v in model_without_ddp.state_dict().items()
        if not k.startswith("patch_head.")
    }


def distributed_sum_scalar(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        y = x.detach().clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y
    return x.detach().clone()


def extract_model_state_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def patch_identity_loss(patch_pred, attn_patch_detached, active_weights, mode="mse", eps=1e-8):
    """
    Keeps the learned patch-head output close to the input attention map.

    Why: support_outside_loss alone can be satisfied by producing a generic
    central cardiac prior. Identity regularization discourages collapse by
    preserving image-specific structure from the original attention map.

    Args:
      patch_pred: [B,N], normalized prediction from patch_head
      attn_patch_detached: [B,N], normalized detached raw attention
      active_weights: [B], usually support_weights
      mode: mse or kl_pred_to_attn
    """
    mode = str(mode).lower().strip()
    w = active_weights.float().clamp_min(0.0)

    if mode == "none":
        return torch.zeros((), dtype=patch_pred.dtype, device=patch_pred.device)

    if mode == "mse":
        per_sample = ((patch_pred - attn_patch_detached) ** 2).mean(dim=-1)
    elif mode in {"kl", "kl_pred_to_attn"}:
        p = patch_pred.clamp_min(eps)
        q = attn_patch_detached.clamp_min(eps)
        per_sample = (p * (p.log() - q.log())).sum(dim=-1)
    else:
        raise ValueError("identity_reg_mode must be one of: mse, kl_pred_to_attn, none")

    return weighted_mean_a6(per_sample, w)
