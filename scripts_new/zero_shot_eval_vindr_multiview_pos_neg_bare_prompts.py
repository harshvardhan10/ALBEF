#!/usr/bin/env python3
"""VinDr-CXR zero-shot classification for learned multi-view ALBEF fusion.

For every VinDr image, this evaluator constructs three spatially aligned views:
    1. original CXR
    2. lung-masked CXR
    3. heart-masked CXR

The learned three-view ALBEF model fuses corresponding ViT tokens and returns
one normalized ITC image feature. For an ordinary label c:

    positive prompt = "c"
    negative prompt = "no c"

The saved classification score is exactly the same bare-prompt probability used
by the existing single-view evaluator:

    softmax([sim(image, negative), sim(image, positive)] / temperature)[positive]

No ITM score or reranking is used during zero-shot classification.

The generated NPZ contains at least the four keys required by
``evaluate_zero_shot_f1_from_validation.py``:
    image_ids, label_names, scores, y_true

It also stores the positive/negative similarities, margins, prompt strings, and
learned temperature for reproducibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml


ALBEF_MEAN = (0.48145466, 0.4578275, 0.40821073)
ALBEF_STD = (0.26862954, 0.26130258, 0.27577711)
VIEW_NAME = "multiview_fusion"


def build_bare_prompt_pair(label: str) -> tuple[str, str]:
    """Return the canonical (positive, negative) bare-prompt pair."""
    clean = str(label).replace("_", " ").strip()
    if not clean:
        raise ValueError("Encountered an empty label name")

    # Keep the same special case as the existing single-view evaluator.
    if clean.casefold() == "no finding":
        return "No finding", "Finding"

    return clean, f"no {clean}"


def compute_positive_probability(
    positive_similarity: torch.Tensor,
    negative_similarity: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Convert +/- ITC similarities to p(positive) using ALBEF temperature."""
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            f"Temperature must be finite and > 0, got {temperature}"
        )
    if positive_similarity.shape != negative_similarity.shape:
        raise ValueError(
            "Positive/negative similarity shapes differ: "
            f"positive={tuple(positive_similarity.shape)}, "
            f"negative={tuple(negative_similarity.shape)}"
        )

    pair_logits = torch.stack(
        (negative_similarity, positive_similarity),
        dim=-1,
    )
    pair_logits = pair_logits / temperature
    return torch.softmax(pair_logits, dim=-1)[..., 1]


def build_output_prefix(checkpoint_path: str | Path) -> str:
    checkpoint_path = Path(checkpoint_path)
    return f"vindr_bare_pair_itc_{VIEW_NAME}_{checkpoint_path.stem}"


def build_image_transform(image_res: int):
    return transforms.Compose(
        [
            transforms.Resize(
                (int(image_res), int(image_res)),
                interpolation=Image.Resampling.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(ALBEF_MEAN, ALBEF_STD),
        ]
    )


class VinDrMultiViewDataset(Dataset):
    """VinDr image plus lung/heart masked views for fused zero-shot inference."""

    def __init__(
        self,
        *,
        labels_csv: str | Path,
        images_root: str | Path,
        lung_mask_root: str | Path,
        heart_mask_root: str | Path,
        image_res: int,
        max_images: int | None = None,
    ) -> None:
        self.labels_csv = Path(labels_csv)
        self.images_root = Path(images_root)
        self.lung_mask_root = Path(lung_mask_root)
        self.heart_mask_root = Path(heart_mask_root)
        self.transform = build_image_transform(image_res)

        if not self.labels_csv.is_file():
            raise FileNotFoundError(f"labels_csv not found: {self.labels_csv}")
        for root, name in (
            (self.images_root, "images_root"),
            (self.lung_mask_root, "lung_mask_root"),
            (self.heart_mask_root, "heart_mask_root"),
        ):
            if not root.is_dir():
                raise FileNotFoundError(f"{name} not found: {root}")

        self.df = pd.read_csv(self.labels_csv)
        if self.df.shape[1] < 2:
            raise ValueError(
                "Labels CSV must contain image_id plus at least one label"
            )

        self.id_col = (
            "image_id" if "image_id" in self.df.columns else self.df.columns[0]
        )
        self.label_cols = [
            column for column in self.df.columns if column != self.id_col
        ]

        if max_images is not None:
            max_images = int(max_images)
            if max_images <= 0:
                raise ValueError("max_images must be positive")
            self.df = self.df.iloc[:max_images]

        self.df = self.df.reset_index(drop=True)
        self.df[self.id_col] = self.df[self.id_col].astype(str)

        if self.df[self.id_col].duplicated().any():
            duplicates = self.df.loc[
                self.df[self.id_col].duplicated(), self.id_col
            ].head(10)
            raise ValueError(
                "Labels CSV contains duplicate image IDs. Examples: "
                + ", ".join(map(str, duplicates.tolist()))
            )

        for label in self.label_cols:
            values = pd.to_numeric(self.df[label], errors="raise")
            unique = set(values.dropna().unique().tolist())
            if not unique.issubset({0, 1, 0.0, 1.0}):
                raise ValueError(
                    f"Label {label!r} must contain only 0/1 values; "
                    f"found {sorted(unique)[:10]}"
                )
            if values.isna().any():
                raise ValueError(f"Label {label!r} contains NaN values")
            self.df[label] = values.astype(np.float32)

        self._verify_files()
        print(
            f"[Data] view={VIEW_NAME} images={len(self.df)} "
            f"labels={len(self.label_cols)}",
            flush=True,
        )

    def _paths(self, image_id: str) -> tuple[Path, Path, Path]:
        image_path = self.images_root / f"{image_id}.png"
        lung_path = self.lung_mask_root / image_id[:2] / f"{image_id}.png"
        heart_path = self.heart_mask_root / image_id[:2] / f"{image_id}.png"
        return image_path, lung_path, heart_path

    def _verify_files(self) -> None:
        missing: list[str] = []
        for image_id in self.df[self.id_col].tolist():
            image_path, lung_path, heart_path = self._paths(str(image_id))
            for path in (image_path, lung_path, heart_path):
                if not path.is_file():
                    missing.append(str(path))
                    if len(missing) >= 10:
                        break
            if len(missing) >= 10:
                break
        if missing:
            raise FileNotFoundError(
                "Missing required VinDr files (first 10):\n" + "\n".join(missing)
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_id = str(row[self.id_col])
        image_path, lung_path, heart_path = self._paths(image_id)

        with Image.open(image_path) as handle:
            original = handle.convert("RGB")
        with Image.open(lung_path) as handle:
            lung_mask = handle.convert("L")
        with Image.open(heart_path) as handle:
            heart_mask = handle.convert("L")

        if original.size != lung_mask.size:
            raise ValueError(
                f"Image/lung-mask size mismatch for {image_id}: "
                f"image={original.size}, mask={lung_mask.size}"
            )
        if original.size != heart_mask.size:
            raise ValueError(
                f"Image/heart-mask size mismatch for {image_id}: "
                f"image={original.size}, mask={heart_mask.size}"
            )

        black = Image.new("RGB", original.size, (0, 0, 0))
        lung = Image.composite(original, black, lung_mask)
        heart = Image.composite(original, black, heart_mask)

        original_tensor = self.transform(original)
        lung_tensor = self.transform(lung)
        heart_tensor = self.transform(heart)
        labels = row[self.label_cols].to_numpy(dtype=np.float32)

        return (
            original_tensor,
            lung_tensor,
            heart_tensor,
            labels,
            image_id,
        )


def _strip_ddp_prefix(state: dict) -> dict:
    if state and all(str(key).startswith("module.") for key in state):
        return {
            str(key)[len("module.") :]: value for key, value in state.items()
        }
    return state


def _interpolate_fused_positional_embeddings(state: dict, model) -> None:
    """Interpolate ViT positional embeddings if config/checkpoint resolution differs."""
    from models.vit import interpolate_pos_embed

    pairs = (
        ("visual_encoder_original.pos_embed", model.visual_encoder_original),
        ("visual_encoder_lung.pos_embed", model.visual_encoder_lung),
        ("visual_encoder_heart.pos_embed", model.visual_encoder_heart),
        ("visual_encoder_original_m.pos_embed", model.visual_encoder_original_m),
        ("visual_encoder_lung_m.pos_embed", model.visual_encoder_lung_m),
        ("visual_encoder_heart_m.pos_embed", model.visual_encoder_heart_m),
    )
    for key, encoder in pairs:
        if key in state:
            state[key] = interpolate_pos_embed(state[key], encoder)


def build_model_and_tokenizer(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    device_name: str,
    text_encoder_override: str | None = None,
):
    """Instantiate the fused ALBEF architecture and load one fused checkpoint."""
    from models.model_pretrain_multiview_fusion import ALBEF
    from models.tokenization_bert import BertTokenizer

    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")

    text_encoder = (
        text_encoder_override
        or config.get("text_encoder")
        or "bert-base-uncased"
    )
    tokenizer = BertTokenizer.from_pretrained(text_encoder)

    # init_deit=False is essential here: all visual weights come from the fused
    # checkpoint, so there is no need to download or initialize DeiT weights.
    model = ALBEF(
        config=config,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        init_deit=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"Invalid checkpoint state: {checkpoint_path}")
    state = _strip_ddp_prefix(state)
    _interpolate_fused_positional_embeddings(state, model)

    message = model.load_state_dict(state, strict=True)
    print(
        f"[Checkpoint] Loaded fused weights from {checkpoint_path}: {message}",
        flush=True,
    )

    device = torch.device(device_name)
    model = model.to(device)
    model.eval()
    return model, tokenizer, config, device


@torch.no_grad()
def encode_bare_prompt_pairs(
    model,
    tokenizer,
    label_names: Sequence[str],
    device: torch.device,
    max_length: int = 32,
):
    """Encode one positive and one negative prompt independently per label."""
    prompt_pairs = [build_bare_prompt_pair(label) for label in label_names]
    positive_prompts = [pair[0] for pair in prompt_pairs]
    negative_prompts = [pair[1] for pair in prompt_pairs]
    all_prompts = positive_prompts + negative_prompts

    tokenized = tokenizer(
        all_prompts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}

    text_output = model.text_encoder.bert(
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
        return_dict=True,
        mode="text",
    )
    text_features = model.text_proj(text_output.last_hidden_state[:, 0, :])
    text_features = F.normalize(text_features, dim=-1)

    num_labels = len(label_names)
    positive_features = text_features[:num_labels]
    negative_features = text_features[num_labels:]
    return (
        positive_features,
        negative_features,
        positive_prompts,
        negative_prompts,
    )


def get_temperature(model, override: float | None = None) -> float:
    """Read the learned ALBEF ITC temperature unless explicitly overridden."""
    if override is not None:
        temperature = float(override)
    elif hasattr(model, "temp"):
        value = model.temp.detach().float().cpu()
        if value.numel() != 1:
            raise ValueError(
                f"Expected scalar model.temp, got shape {tuple(value.shape)}"
            )
        temperature = float(value.item())
    else:
        raise AttributeError(
            "This model has no scalar 'temp'. Pass --temperature explicitly."
        )

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            f"Temperature must be finite and > 0, got {temperature}"
        )
    return temperature


def safe_auc(y_true: np.ndarray, scores: np.ndarray):
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    label_names: Sequence[str],
) -> dict:
    """Compute threshold-independent AUC metrics for the saved score matrix."""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.shape != scores.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, scores={scores.shape}"
        )
    if y_true.shape[1] != len(label_names):
        raise ValueError(
            f"Expected {len(label_names)} score columns, got {y_true.shape[1]}"
        )

    per_label_auc = {}
    per_label_support = {}
    aucs = []
    for column, label in enumerate(label_names):
        target = y_true[:, column]
        auc = safe_auc(target, scores[:, column])
        per_label_auc[str(label)] = auc
        per_label_support[str(label)] = int(target.sum())
        if auc is not None:
            aucs.append(auc)

    return {
        "per_label_auc": per_label_auc,
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "micro_auc": safe_auc(y_true.ravel(), scores.ravel()),
        "per_label_support": per_label_support,
    }


def save_npz(
    output_path: str | Path,
    image_ids,
    label_names,
    positive_prompts,
    negative_prompts,
    y_true,
    positive_similarities,
    negative_similarities,
    margins,
    scores,
    temperature,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image_ids=np.asarray(image_ids, dtype=object),
        label_names=np.asarray(label_names, dtype=object),
        positive_prompts=np.asarray(positive_prompts, dtype=object),
        negative_prompts=np.asarray(negative_prompts, dtype=object),
        y_true=np.asarray(y_true, dtype=np.float32),
        positive_similarities=np.asarray(
            positive_similarities, dtype=np.float32
        ),
        negative_similarities=np.asarray(
            negative_similarities, dtype=np.float32
        ),
        margins=np.asarray(margins, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
        temperature=np.asarray(temperature, dtype=np.float32),
    )


def evaluate_checkpoint(args, checkpoint_path: str | Path) -> dict:
    checkpoint_path = Path(checkpoint_path)
    print(f"\n========== {checkpoint_path} ==========", flush=True)

    model, tokenizer, config, device = build_model_and_tokenizer(
        config_path=args.config,
        checkpoint_path=checkpoint_path,
        device_name=args.device,
        text_encoder_override=args.text_encoder,
    )

    temperature = get_temperature(model, args.temperature)
    print(f"[ITC] learned/used temperature={temperature:.8f}", flush=True)

    dataset = VinDrMultiViewDataset(
        labels_csv=args.labels_csv,
        images_root=args.images_root,
        lung_mask_root=args.lung_mask_root,
        heart_mask_root=args.heart_mask_root,
        image_res=int(config["image_res"]),
        max_images=args.max_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    (
        positive_text,
        negative_text,
        positive_prompts,
        negative_prompts,
    ) = encode_bare_prompt_pairs(
        model,
        tokenizer,
        dataset.label_cols,
        device,
        max_length=args.max_text_length,
    )

    for label, positive, negative in zip(
        dataset.label_cols,
        positive_prompts,
        negative_prompts,
    ):
        print(
            f"[Prompt] {label!r}: positive={positive!r}, "
            f"negative={negative!r}",
            flush=True,
        )

    all_positive = []
    all_negative = []
    all_scores = []
    all_labels = []
    all_ids: list[str] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            (
                original,
                lung,
                heart,
                labels,
                image_ids,
            ) = batch

            original = original.to(device, non_blocking=True)
            lung = lung.to(device, non_blocking=True)
            heart = heart.to(device, non_blocking=True)

            image_features = model.get_image_features(
                original,
                lung,
                heart,
            )
            positive_sim = image_features @ positive_text.t()
            negative_sim = image_features @ negative_text.t()
            positive_probability = compute_positive_probability(
                positive_sim,
                negative_sim,
                temperature,
            )

            all_positive.append(positive_sim.cpu().numpy())
            all_negative.append(negative_sim.cpu().numpy())
            all_scores.append(positive_probability.cpu().numpy())
            all_labels.append(labels.numpy())
            all_ids.extend(map(str, image_ids))

            if batch_index % 10 == 0 or batch_index == len(loader):
                print(
                    f"[Eval] processed {batch_index}/{len(loader)} batches",
                    flush=True,
                )

    positive_similarities = np.vstack(all_positive).astype(np.float32)
    negative_similarities = np.vstack(all_negative).astype(np.float32)
    scores = np.vstack(all_scores).astype(np.float32)
    y_true = np.vstack(all_labels).astype(np.float32)
    margins = positive_similarities - negative_similarities

    if len(all_ids) != len(dataset):
        raise RuntimeError(
            f"Expected {len(dataset)} evaluated images, got {len(all_ids)}"
        )
    if scores.shape != y_true.shape:
        raise RuntimeError(
            f"Saved scores/y_true shape mismatch: {scores.shape} vs {y_true.shape}"
        )

    metrics = compute_metrics(y_true, scores, dataset.label_cols)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = build_output_prefix(checkpoint_path)
    scores_path = output_dir / f"{prefix}_scores.npz"
    metrics_path = output_dir / f"{prefix}_metrics.json"

    save_npz(
        scores_path,
        all_ids,
        dataset.label_cols,
        positive_prompts,
        negative_prompts,
        y_true,
        positive_similarities,
        negative_similarities,
        margins,
        scores,
        temperature,
    )

    result = {
        "checkpoint": str(checkpoint_path),
        "view_type": VIEW_NAME,
        "num_images": int(len(all_ids)),
        "label_names": list(map(str, dataset.label_cols)),
        "positive_prompts": dict(
            zip(map(str, dataset.label_cols), positive_prompts)
        ),
        "negative_prompts": dict(
            zip(map(str, dataset.label_cols), negative_prompts)
        ),
        "scoring": (
            "softmax([negative_similarity, positive_similarity] / "
            "temperature)[1]"
        ),
        "uses_itm": False,
        "uses_multiview_fusion": True,
        "temperature": temperature,
        "scores_file": str(scores_path),
        "classification": metrics,
    }
    with metrics_path.open("w") as handle:
        json.dump(result, handle, indent=2)

    print(f"[Result] macro AUC: {metrics['macro_auc']}", flush=True)
    print(f"[Result] saved: {scores_path}", flush=True)
    print(f"[Result] saved: {metrics_path}", flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Learned multi-view ALBEF VinDr zero-shot evaluation with "
            "bare positive/negative ITC prompts"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--lung_mask_root", required=True)
    parser.add_argument("--heart_mask_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--max_text_length", type=int, default=32)
    parser.add_argument(
        "--text_encoder",
        default=None,
        help=(
            "Optional tokenizer/BERT source override. By default uses "
            "config['text_encoder'] or bert-base-uncased."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override model.temp; by default use the learned checkpoint value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = {}
    for checkpoint in args.checkpoints:
        checkpoint_path = Path(checkpoint)
        results[checkpoint_path.name] = evaluate_checkpoint(
            args,
            checkpoint_path,
        )

    combined_path = Path(args.output_dir) / (
        f"vindr_bare_pair_itc_{VIEW_NAME}_all_checkpoints.json"
    )
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"[Result] saved: {combined_path}", flush=True)


if __name__ == "__main__":
    main()
