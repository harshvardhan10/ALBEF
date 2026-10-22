"""Original anatomy helpers; workflow-specific defaults remain explicit."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def build_support_weights_from_captions(text, config, device):
    """
    Builds per-sample weights for anatomy support loss.

    Modified it for compatibility with both Cardiomegaly and PE

    Disease-agnostic modes:
        "none"
        "all_target_captions"
        "uncertainty_weighted"
        "positive_only"

    For current experiments:
        anatomy_target_phrase: "cardiomegaly"
        anatomy_target_phrase: "pleural effusion"
    """

    support_mode = config.get("support_mode", "all_target_captions")
    support_mode = str(support_mode).lower().strip()

    target_phrase = str(config.get("anatomy_target_phrase", "")).lower().strip()

    # ------------------------------------------------------------------
    # Backward compatibility with Cardiomegaly YAMLs
    # ------------------------------------------------------------------
    if support_mode == "all_cardiomegaly_captions":
        support_mode = "all_target_captions"

    if support_mode != "none" and target_phrase == "":
        raise ValueError(
            "anatomy_target_phrase must be set when support_mode is not 'none'."
        )

    positive_weight = float(config.get("positive_caption_weight", 1.0))
    uncertain_weight = float(config.get("uncertain_caption_weight", 0.5))

    weights = []

    for t in text:
        t_lower = str(t).lower().strip()

        exact_positive = t_lower == target_phrase
        contains_target = target_phrase in t_lower
        uncertain_target = contains_target and ("uncertain" in t_lower)

        if support_mode == "none":
            weight = 0.0

        elif support_mode == "all_target_captions":
            weight = 1.0 if contains_target else 0.0

        elif support_mode == "uncertainty_weighted":
            if exact_positive:
                weight = positive_weight
            elif uncertain_target:
                weight = uncertain_weight
            else:
                weight = 0.0

        elif support_mode == "positive_only":
            weight = 1.0 if exact_positive else 0.0

        else:
            raise ValueError(
                f"Unknown support_mode: {support_mode}. "
                "Expected one of: none, all_target_captions, "
                "uncertainty_weighted, positive_only, "
                "all_cardiomegaly_captions."
            )

        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float, device=device)


def build_dummy_anatomy_prior_mask(target_phrase, batch_size, height, width, device):
    """
    Builds fixed dummy anatomy masks for anatomy-aware support regularization.

    Cardiomegaly:
        central lower chest / heart region.

    Pleural Effusion:
        bilateral lower lung-base regions near the costophrenic angles.
    """

    target_phrase = str(target_phrase).lower().strip()

    prior_mask = torch.zeros(
        (batch_size, 1, height, width),
        dtype=torch.float32,
        device=device,
    )

    if target_phrase == "cardiomegaly":
        # Central cardiac silhouette proxy
        h1, h2 = int(0.35 * height), int(0.75 * height)
        w1, w2 = int(0.25 * width), int(0.75 * width)

        prior_mask[:, :, h1:h2, w1:w2] = 1.0

    elif target_phrase == "pleural effusion":
        # Bilateral lower pleural/lung-base proxy
        h1, h2 = int(0.55 * height), int(0.93 * height)

        # Right lower lung in image coordinates
        w1_r, w2_r = int(0.07 * width), int(0.45 * width)

        # Left lower lung in image coordinates
        w1_l, w2_l = int(0.55 * width), int(0.93 * width)

        prior_mask[:, :, h1:h2, w1_r:w2_r] = 1.0
        prior_mask[:, :, h1:h2, w1_l:w2_l] = 1.0

    else:
        raise ValueError(
            f"No dummy anatomy prior mask defined for anatomy_target_phrase='{target_phrase}'. "
            "Supported dummy masks: 'cardiomegaly', 'pleural effusion'."
        )

    return prior_mask


def load_external_anatomy_prior_template(prior_path, device):
    """
    Load a dataset-level anatomy prior saved as .npy/.npz and return shape [1, 1, H, W].

    Expected use for A3-heartseg:
      anatomy_prior_path: anatomy_priors/vindr_train_heart_prior_256.npy

    The prior can be binary or soft. Values are clamped/normalized to [0, 1].
    """
    prior_path = Path(prior_path)
    if not prior_path.exists():
        raise FileNotFoundError(f"External anatomy prior not found: {prior_path}")

    if prior_path.suffix == ".npy":
        arr = np.load(prior_path)
    elif prior_path.suffix == ".npz":
        data = np.load(prior_path)
        if "prior" in data.files:
            arr = data["prior"]
        elif "mask" in data.files:
            arr = data["mask"]
        else:
            arr = data[data.files[0]]
    else:
        raise ValueError(f"Unsupported prior file extension: {prior_path.suffix}")

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

    # Accept HxW, 1xHxW, or 1x1xHxW.
    if arr.ndim == 2:
        arr = arr[None, None, :, :]
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[None, :, :, :]
        else:
            raise ValueError(f"Expected prior shape HxW, 1xHxW, or 1x1xHxW; got {arr.shape}")
    elif arr.ndim == 4:
        if arr.shape[0] != 1 or arr.shape[1] != 1:
            raise ValueError(f"Expected prior shape 1x1xHxW; got {arr.shape}")
    else:
        raise ValueError(f"Expected prior shape HxW, 1xHxW, or 1x1xHxW; got {arr.shape}")

    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max <= 0:
        raise ValueError(f"Loaded prior is all zero: {prior_path}")

    # If values are not already in [0, 1], min-max normalize.
    if arr_min < 0.0 or arr_max > 1.0:
        arr = (arr - arr_min) / max(arr_max - arr_min, 1e-8)

    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr).float().to(device)


def build_external_anatomy_prior_mask(
    prior_template,
    batch_size,
    height,
    width,
    device,
    binarize_threshold=None,
):
    """
    Resize a loaded dataset-level prior to the current image tensor size and repeat over batch.
    Returns [B, 1, H, W].
    """
    prior = prior_template.to(device=device, dtype=torch.float32)

    if prior.shape[-2:] != (height, width):
        prior = F.interpolate(
            prior,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    prior = prior.clamp(0.0, 1.0)

    if binarize_threshold is not None:
        prior = (prior >= float(binarize_threshold)).float()

    return prior.expand(batch_size, 1, height, width).contiguous()


def build_anatomy_prior_mask(target_phrase, batch_size, height, width, device, config, prior_template=None):
    """
    Build the anatomy prior used by the support loss.

    If config['anatomy_prior_path'] is set, use the external segmentation-derived prior.
    Otherwise, fall back to the original dummy rectangle/lung-base prior.
    """
    if prior_template is not None:
        return build_external_anatomy_prior_mask(
            prior_template=prior_template,
            batch_size=batch_size,
            height=height,
            width=width,
            device=device,
            binarize_threshold=config.get("anatomy_prior_binarize_threshold", None),
        )

    return build_dummy_anatomy_prior_mask(
        target_phrase=target_phrase,
        batch_size=batch_size,
        height=height,
        width=width,
        device=device,
    )


def unwrap_model(model):
    """Return the underlying module if model is wrapped in DDP/DataParallel."""
    return model.module if hasattr(model, "module") else model


def strip_module_prefix(state_dict):
    """Make checkpoints saved with DistributedDataParallel compatible with normal models."""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_state_dict_flexible(model, checkpoint_path, device="cpu", interpolate_pos=True):
    """
    Load a checkpoint into an ALBEF model. Supports checkpoints saved as:
        {'model': state_dict, ...}, {'state_dict': state_dict}, or raw state_dict.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = strip_module_prefix(state_dict)

    if interpolate_pos:
        from models.vit import interpolate_pos_embed

        if "visual_encoder.pos_embed" in state_dict:
            state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
                state_dict["visual_encoder.pos_embed"], model.visual_encoder
            )
        if "visual_encoder_m.pos_embed" in state_dict and hasattr(model, "visual_encoder_m"):
            state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
                state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m
            )

    msg = model.load_state_dict(state_dict, strict=False)
    return msg


def get_global_image_text_features(model, image, text_input):
    """
    Extract normalized global image and text features from an ALBEF-style model.

    These are the global image/text representations used to build a batch-wise
    image-text similarity matrix for classification-preservation distillation.
    """
    model = unwrap_model(model)

    image_embeds = model.visual_encoder(image)
    if isinstance(image_embeds, (tuple, list)):
        image_embeds = image_embeds[0]

    image_feat = F.normalize(
        model.vision_proj(image_embeds[:, 0, :]),
        dim=-1,
    )

    if hasattr(model.text_encoder, "bert"):
        text_output = model.text_encoder.bert(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            return_dict=True,
            mode="text",
        )
    else:
        try:
            text_output = model.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                return_dict=True,
                mode="text",
            )
        except TypeError:
            text_output = model.text_encoder(
                input_ids=text_input.input_ids,
                attention_mask=text_input.attention_mask,
                return_dict=True,
            )

    text_feat = F.normalize(
        model.text_proj(text_output.last_hidden_state[:, 0, :]),
        dim=-1,
    )

    return image_feat, text_feat


def classification_preservation_loss(
    student_model,
    teacher_model,
    image,
    text_input,
    temperature=0.07,
):
    """
    Preserve the A0 teacher's global image-text matching behaviour.

    For each batch, the frozen A0 teacher produces image-to-text and text-to-image
    similarity distributions. The A4 student is penalized if its similarity
    distributions drift away from the teacher.
    """
    student_image_feat, student_text_feat = get_global_image_text_features(
        student_model,
        image,
        text_input,
    )

    with torch.no_grad():
        teacher_image_feat, teacher_text_feat = get_global_image_text_features(
            teacher_model,
            image,
            text_input,
        )

    student_sim_i2t = student_image_feat @ student_text_feat.t()
    student_sim_t2i = student_text_feat @ student_image_feat.t()

    teacher_sim_i2t = teacher_image_feat @ teacher_text_feat.t()
    teacher_sim_t2i = teacher_text_feat @ teacher_image_feat.t()

    temperature = float(temperature)

    teacher_prob_i2t = F.softmax(teacher_sim_i2t / temperature, dim=-1)
    teacher_prob_t2i = F.softmax(teacher_sim_t2i / temperature, dim=-1)

    student_log_prob_i2t = F.log_softmax(student_sim_i2t / temperature, dim=-1)
    student_log_prob_t2i = F.log_softmax(student_sim_t2i / temperature, dim=-1)

    loss_i2t = F.kl_div(
        student_log_prob_i2t,
        teacher_prob_i2t,
        reduction="batchmean",
    )
    loss_t2i = F.kl_div(
        student_log_prob_t2i,
        teacher_prob_t2i,
        reduction="batchmean",
    )

    return 0.5 * (loss_i2t + loss_t2i)


def build_support_weights_a6(text, config, device):
    """
    Builds per-sample weights for anatomy support loss.

    For A6.1 use:
      support_mode: positive_only
      anatomy_target_phrase: cardiomegaly
    """
    support_mode = str(config.get("support_mode", "positive_only")).lower().strip()
    target_phrase = str(config.get("anatomy_target_phrase", "")).lower().strip()

    if support_mode == "all_cardiomegaly_captions":
        support_mode = "all_target_captions"

    if support_mode != "none" and target_phrase == "":
        raise ValueError("anatomy_target_phrase must be set when support_mode is not 'none'.")

    positive_weight = float(config.get("positive_caption_weight", 1.0))
    uncertain_weight = float(config.get("uncertain_caption_weight", 0.5))

    weights = []
    for t in text:
        t_lower = str(t).lower().strip()
        exact_positive = t_lower == target_phrase
        contains_target = target_phrase in t_lower
        uncertain_target = contains_target and ("uncertain" in t_lower)

        if support_mode == "none":
            weight = 0.0
        elif support_mode == "all_target_captions":
            weight = 1.0 if contains_target else 0.0
        elif support_mode == "uncertainty_weighted":
            if exact_positive:
                weight = positive_weight
            elif uncertain_target:
                weight = uncertain_weight
            else:
                weight = 0.0
        elif support_mode == "positive_only":
            weight = 1.0 if exact_positive else 0.0
        else:
            raise ValueError(
                f"Unknown support_mode={support_mode}. Expected: none, all_target_captions, "
                "uncertainty_weighted, positive_only."
            )
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_dummy_anatomy_prior_mask_a5(target_phrase, batch_size, height, width, device):
    """
    Fixed anatomy prior proxies.

    Cardiomegaly:
        central cardiac silhouette proxy.
    Pleural Effusion:
        bilateral lower lung-base proxy.
    """
    target_phrase = str(target_phrase).lower().strip()

    prior_mask = torch.zeros(
        (batch_size, 1, height, width),
        dtype=torch.float32,
        device=device,
    )

    if target_phrase == "cardiomegaly":
        h1, h2 = int(0.35 * height), int(0.75 * height)
        w1, w2 = int(0.25 * width), int(0.75 * width)
        prior_mask[:, :, h1:h2, w1:w2] = 1.0

    elif target_phrase == "pleural effusion":
        h1, h2 = int(0.55 * height), int(0.93 * height)
        w1_r, w2_r = int(0.07 * width), int(0.45 * width)
        w1_l, w2_l = int(0.55 * width), int(0.93 * width)
        prior_mask[:, :, h1:h2, w1_r:w2_r] = 1.0
        prior_mask[:, :, h1:h2, w1_l:w2_l] = 1.0

    else:
        raise ValueError(
            f"No dummy anatomy prior mask defined for anatomy_target_phrase='{target_phrase}'. "
            "Supported dummy masks: 'cardiomegaly', 'pleural effusion'."
        )

    return prior_mask


def build_dummy_anatomy_prior_mask_a6(target_phrase, batch_size, height, width, device):
    """
    Same fixed anatomy prior as previous experiments.
    Cardiomegaly: central cardiac silhouette proxy.
    """
    target_phrase = str(target_phrase).lower().strip()
    prior_mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32, device=device)

    if target_phrase == "cardiomegaly":
        h1, h2 = int(0.35 * height), int(0.75 * height)
        w1, w2 = int(0.25 * width), int(0.75 * width)
        prior_mask[:, :, h1:h2, w1:w2] = 1.0
    elif target_phrase == "pleural effusion":
        h1, h2 = int(0.55 * height), int(0.93 * height)
        w1_r, w2_r = int(0.07 * width), int(0.45 * width)
        w1_l, w2_l = int(0.55 * width), int(0.93 * width)
        prior_mask[:, :, h1:h2, w1_r:w2_r] = 1.0
        prior_mask[:, :, h1:h2, w1_l:w2_l] = 1.0
    else:
        raise ValueError(
            f"No dummy anatomy prior mask defined for anatomy_target_phrase='{target_phrase}'. "
            "Supported: cardiomegaly, pleural effusion."
        )

    return prior_mask
