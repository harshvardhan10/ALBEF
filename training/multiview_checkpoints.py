"""Original initialization/resumption helpers shared by all six fusion trainers."""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import torch
from evaluation.checkpoint_roles import DualAUCSelectionState


def interpolate_pos_embed(state, encoder):
    from models.vit import interpolate_pos_embed as interpolate
    return interpolate(state, encoder)

def raw_model(model):
    return model.module if hasattr(model, "module") else model


def _load_checkpoint_state(path: str | Path) -> tuple[dict, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"Invalid checkpoint state at {path}")

    # Be tolerant of checkpoints saved directly from DDP.
    if state and all(str(key).startswith("module.") for key in state):
        state = {
            str(key)[len("module.") :]: value
            for key, value in state.items()
        }

    config = checkpoint.get("config", {})
    return state, config


def _substate(state: dict, prefix: str) -> dict:
    prefix_dot = prefix + "."
    result = {
        key[len(prefix_dot) :]: value
        for key, value in state.items()
        if key.startswith(prefix_dot)
    }
    if not result:
        raise KeyError(f"No parameters found for prefix {prefix!r}")
    return result


def _load_single_view_visual_encoder(
    *,
    target_encoder,
    checkpoint_path: str | Path,
    view_name: str,
) -> None:
    state, _ = _load_checkpoint_state(checkpoint_path)
    visual_state = _substate(state, "visual_encoder")

    if "pos_embed" in visual_state:
        visual_state["pos_embed"] = interpolate_pos_embed(
            visual_state["pos_embed"],
            target_encoder,
        )

    msg = target_encoder.load_state_dict(
        visual_state,
        strict=True,
    )
    print(
        f"[Init] {view_name} visual encoder <- {checkpoint_path}: {msg}",
        flush=True,
    )


def _load_shared_from_single_view_checkpoint(
    *,
    model: ALBEF,
    checkpoint_path: str | Path,
    source_name: str,
) -> None:
    state, _ = _load_checkpoint_state(checkpoint_path)

    module_map = {
        "text_encoder": model.text_encoder,
        "vision_proj": model.vision_proj,
        "text_proj": model.text_proj,
        "itm_head": model.itm_head,
    }
    for prefix, module in module_map.items():
        msg = module.load_state_dict(
            _substate(state, prefix),
            strict=True,
        )
        print(
            f"[Init shared/{source_name}] {prefix}: {msg}",
            flush=True,
        )

    if "temp" not in state:
        raise KeyError(
            f"{checkpoint_path} does not contain learned temperature 'temp'"
        )
    with torch.no_grad():
        model.temp.copy_(state["temp"].reshape_as(model.temp))

    print(
        f"[Init shared/{source_name}] temp={float(model.temp):.8f}",
        flush=True,
    )


def initialize_from_three_single_view_checkpoints(
    *,
    model: ALBEF,
    original_checkpoint: str | Path,
    lung_checkpoint: str | Path,
    heart_checkpoint: str | Path,
    shared_components_from: str = "original",
) -> None:
    checkpoints = {
        "original": Path(original_checkpoint),
        "lung": Path(lung_checkpoint),
        "heart": Path(heart_checkpoint),
    }

    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    _load_single_view_visual_encoder(
        target_encoder=model.visual_encoder_original,
        checkpoint_path=checkpoints["original"],
        view_name="original",
    )
    _load_single_view_visual_encoder(
        target_encoder=model.visual_encoder_lung,
        checkpoint_path=checkpoints["lung"],
        view_name="lung",
    )
    _load_single_view_visual_encoder(
        target_encoder=model.visual_encoder_heart,
        checkpoint_path=checkpoints["heart"],
        view_name="heart",
    )

    shared_components_from = str(shared_components_from).lower().strip()
    if shared_components_from not in checkpoints:
        raise ValueError(
            "shared_components_from must be original, lung, or heart"
        )

    _load_shared_from_single_view_checkpoint(
        model=model,
        checkpoint_path=checkpoints[shared_components_from],
        source_name=shared_components_from,
    )

    # The anchored cross-attention fusion module keeps its fresh initialization.
    # Momentum copies must now mirror the newly assembled online network.
    model.reset_momentum_from_online()

    # image_queue/text_queue intentionally remain fresh random normalized
    # buffers from __init__, because old queues contain single-view features.
    model.queue_ptr.zero_()

    print(
        "[Init] Three-view model assembled; momentum reset; "
        "single-view queues were not reused.",
        flush=True,
    )


def _interpolate_fused_positional_embeddings(
    state: dict,
    model: ALBEF,
) -> None:
    pairs = (
        ("visual_encoder_original.pos_embed", model.visual_encoder_original),
        ("visual_encoder_lung.pos_embed", model.visual_encoder_lung),
        ("visual_encoder_heart.pos_embed", model.visual_encoder_heart),
        (
            "visual_encoder_original_m.pos_embed",
            model.visual_encoder_original_m,
        ),
        ("visual_encoder_lung_m.pos_embed", model.visual_encoder_lung_m),
        ("visual_encoder_heart_m.pos_embed", model.visual_encoder_heart_m),
    )

    for key, encoder in pairs:
        if key in state:
            state[key] = interpolate_pos_embed(
                state[key],
                encoder,
            )


def load_fused_weights_only(
    model: ALBEF,
    path: str | Path,
) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    if state and all(str(key).startswith("module.") for key in state):
        state = {
            str(key)[len("module.") :]: value
            for key, value in state.items()
        }

    _interpolate_fused_positional_embeddings(state, model)
    msg = model.load_state_dict(state, strict=True)
    print(
        f"[Checkpoint] Loaded fused model weights from {path}: {msg}",
        flush=True,
    )


def resume_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
) -> Tuple[int, float, DualAUCSelectionState]:
    checkpoint = torch.load(path, map_location="cpu")
    required = {"model", "optimizer", "lr_scheduler"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            f"Cannot resume from {path}: missing {sorted(missing)}. "
            "Resume from checkpoint_last.pth."
        )

    state = checkpoint["model"]
    _interpolate_fused_positional_embeddings(state, model)
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["lr_scheduler"])

    selection_state = DualAUCSelectionState.from_dict(
        checkpoint.get("validation_selection_state")
    )
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        float(checkpoint.get("best_loss", float("inf"))),
        selection_state,
    )


def _resolve_initial_checkpoint(
    args_value: str,
    config: dict,
    view_name: str,
) -> str:
    if args_value:
        return args_value

    init_cfg = config.get("initial_checkpoints", {})
    value = init_cfg.get(view_name, "")
    return str(value) if value else ""
