"""
Contains:
- Config + model + tokenizer builder
- Image transforms
- Prompt construction + multi-prompt text embeddings
- Image CLS embeddings
"""

from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms

from models.model_pretrain import ALBEF
from models.tokenization_bert import BertTokenizer


# ==========================
# Config / Model / Tokenizer
# ==========================

def load_config(config_path):
    """Load YAML config used for ALBEF training."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model_and_tokenizer(
    config_path,
    ckpt_path,
    device = "cuda",
):
    """
    Build ALBEF + tokenizer and load checkpoint

    Returns:
        model, tokenizer, config, device_torch
    """
    config = load_config(config_path)

    # Device
    device_torch = torch.device(device if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("[Model] Building ALBEF...")
    model = ALBEF(
        config=config,
        text_encoder="bert-base-uncased",
        tokenizer=tokenizer,
        init_deit=False,   # load checkpoint
    )

    ckpt_path = Path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    msg = model.load_state_dict(state_dict, strict=False)
    print("[Model] State dict loaded:", msg)

    model = model.to(device_torch)
    model.eval()

    return model, tokenizer, config, device_torch


# ==========================
# Image transforms
# ==========================

def get_image_transform(image_size: int) -> transforms.Compose:

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


# ==========================
# Prompts & Text Embeddings
# ==========================

def build_prompts_for_label(label):
    """
    Multi-prompt template expansion for a single VinDr label.

    Special case:
      - For "No finding" only use the bare label, because
        templates like "There is evidence of No finding" don't make sense.
    """
    clean_label = label.replace("_", " ")

    if clean_label.strip().lower() == "no finding":
        return ["No finding"]

    templates = [
        "{label}",
        "A chest X-ray showing {label}.",
        "Chest radiograph demonstrating {label}.",
        "There is evidence of {label}.",
        "Findings are consistent with {label}.",
    ]
    return [t.format(label=clean_label) for t in templates]


def get_label_text_embeddings(
    model,
    tokenizer,
    labels,
    device,
    max_length,
):
    """
    Compute one embedding per label by averaging over multiple prompts.

    Returns:
        label_embs: torch.Tensor of shape (L, D)
    """
    all_prompts = []
    label_ranges = []  # (start_idx, end_idx) in all_prompts for each label

    for label in labels:
        prompts = build_prompts_for_label(label)
        start = len(all_prompts)
        all_prompts.extend(prompts)
        end = len(all_prompts)
        label_ranges.append((start, end))

    print(f"[Text] Total prompts: {len(all_prompts)} for {len(labels)} labels")

    tokenized = tokenizer(
        all_prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    with torch.no_grad():
        text_output = model.text_encoder.bert(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            return_dict=True,
            mode="text",
        )
        cls = text_output.last_hidden_state[:, 0, :]  # (P, 768)
        feats = model.text_proj(cls)                  # (P, D)
        feats = F.normalize(feats, dim=-1)           # (P, D)

    # Average prompts per label
    label_embs = []
    for (start, end) in label_ranges:
        label_embs.append(feats[start:end].mean(dim=0))
    label_embs = torch.stack(label_embs, dim=0)      # (L, D)

    return label_embs


# ==========================
# Image Embeddings (CLS)
# ==========================

def get_image_embeddings(model: ALBEF, images: torch.Tensor) -> torch.Tensor:
    """
    images: (B, 3, H, W) on device
    returns: (B, D) normalized image embeddings (CLS)
    """
    with torch.no_grad():
        image_embeds = model.visual_encoder(images)   # (B, num_patches+1, 768)
        image_cls = image_embeds[:, 0, :]             # (B, 768)
        image_feat = model.vision_proj(image_cls)     # (B, D)
        image_feat = torch.nn.functional.normalize(image_feat, dim=-1)
    return image_feat


def get_label_text_inputs(tokenizer, labels, max_length):
    """
    Build tokenized text inputs for each label prompt.
    Returns:
        input_ids_dict:  label -> input_ids tensor (1, T)
        attention_mask_dict: label -> attention_mask tensor (1, T)
    """
    input_ids_dict = {}
    attention_mask_dict = {}
    token_mask_dict = {}

    def make_prompt(label):
        return f"This chest X-ray shows {label.lower()}."

    for label in labels:
        text = make_prompt(label)
        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids_dict[label] = enc["input_ids"]
        attention_mask_dict[label] = enc["attention_mask"]

        token_mask = build_text_token_mask(
            tokenizer=tokenizer,
            label=label,
            input_ids=input_ids_dict[label],
            attention_mask=attention_mask_dict[label],
        )  # (T,)
        token_mask_dict[label] = token_mask

    return input_ids_dict, attention_mask_dict, token_mask_dict


def build_text_token_mask(
    tokenizer,
    label: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Build a 1D mask over text tokens for Grad-CAM aggregation.

    - label: e.g. "Cardiomegaly"
    - input_ids: (1, T)
    - attention_mask: (1, T)

    Returns:
        mask: (T,) with 1.0 on the sub-token positions corresponding to the label word(s),
              0.0 elsewhere. If no match is found, falls back to attention_mask (no padding).
    """
    # Tokenize the label in isolation to get its subword sequence
    label_tokens = tokenizer.tokenize(label.lower())
    if not label_tokens:
        # If something goes wrong, just use attention_mask (no padding)
        return attention_mask[0].float()

    # Convert the prompt's IDs back to tokens
    prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    T = len(prompt_tokens)
    L = len(label_tokens)
    mask = torch.zeros(T, dtype=torch.float)

    # Try to find the label_tokens as a contiguous subsequence in the prompt_tokens
    found = False
    for i in range(T - L + 1):
        if prompt_tokens[i : i + L] == label_tokens:
            mask[i : i + L] = 1.0
            found = True
            break

    if not found:
        # Fallback: use attention_mask (ignore padding, treat all non-pad as relevant)
        mask = attention_mask[0].float()

    return mask
