#!/usr/bin/env python3
"""VinDr-CXR zero-shot classification with bare positive/negative ITC prompts.

For an ordinary label c:
    positive prompt = "c"
    negative prompt = "no c"

The prediction saved as ``scores`` is:
    softmax([sim(image, negative), sim(image, positive)] / temperature)[positive]

No ITM fusion, ITM head, or retrieval reranking is used.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src import build_model_and_tokenizer, get_image_embeddings, get_image_transform


def build_bare_prompt_pair(label):
    """Return the canonical (positive, negative) bare-prompt pair."""
    clean = str(label).replace("_", " ").strip()
    if not clean:
        raise ValueError("Encountered an empty label name")

    # "no No finding" is semantically invalid.  Keep this special case explicit.
    if clean.casefold() == "no finding":
        return "No finding", "Finding"

    return clean, f"no {clean}"


@torch.no_grad()
def encode_bare_prompt_pairs(model, tokenizer, label_names, device, max_length=32):
    """Encode one positive and one negative prompt independently per label."""
    prompt_pairs = [build_bare_prompt_pair(label) for label in label_names]
    positive_prompts = [pair[0] for pair in prompt_pairs]
    negative_prompts = [pair[1] for pair in prompt_pairs]
    all_prompts = positive_prompts + negative_prompts

    tokenized = tokenizer(
        all_prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
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
    return positive_features, negative_features, positive_prompts, negative_prompts


def get_temperature(model, override=None):
    """Read ALBEF's learned ITC temperature, unless explicitly overridden."""
    if override is not None:
        temperature = float(override)
    elif hasattr(model, "temp"):
        value = model.temp.detach().float().cpu()
        if value.numel() != 1:
            raise ValueError(f"Expected scalar model.temp, got shape {tuple(value.shape)}")
        temperature = float(value.item())
    else:
        raise AttributeError(
            "This model has no scalar 'temp'. Pass --temperature explicitly."
        )

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Temperature must be finite and > 0, got {temperature}")
    return temperature


class VinDrDataset(Dataset):
    def __init__(
        self,
        labels_csv,
        images_root,
        transform,
        max_images=None,
        view_type="original",
        mask_root=None,
    ):
        self.df = pd.read_csv(labels_csv)
        if self.df.shape[1] < 2:
            raise ValueError("Labels CSV must contain image_id plus at least one label")

        self.id_col = self.df.columns[0]
        self.label_cols = list(self.df.columns[1:])
        self.images_root = Path(images_root)
        self.transform = transform
        self.view_type = view_type
        self.mask_root = Path(mask_root) if mask_root else None

        if max_images is not None:
            self.df = self.df.iloc[:max_images]
        self.df = self.df.reset_index(drop=True)

        if view_type != "original" and self.mask_root is None:
            raise ValueError("--mask_root is required for lung and heart views")

        missing = []
        for image_id in self.df[self.id_col].astype(str):
            image_path = self.images_root / f"{image_id}.png"
            if not image_path.exists():
                missing.append(str(image_path))
            if self.mask_root is not None:
                mask_path = self.mask_root / image_id[:2] / f"{image_id}.png"
                if not mask_path.exists():
                    missing.append(str(mask_path))
            if len(missing) >= 10:
                break
        if missing:
            raise FileNotFoundError("Missing required files (first 10):\n" + "\n".join(missing))

        print(
            f"[Data] view={view_type} images={len(self.df)} "
            f"labels={len(self.label_cols)}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_id = str(row[self.id_col])
        image_path = self.images_root / f"{image_id}.png"
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")

        if self.view_type != "original":
            mask_path = self.mask_root / image_id[:2] / f"{image_id}.png"
            with Image.open(mask_path) as handle:
                mask = handle.convert("L")
            if image.size != mask.size:
                raise ValueError(
                    f"Image/mask size mismatch for {image_id}: "
                    f"image={image.size}, mask={mask.size}"
                )
            image = Image.composite(image, Image.new("RGB", image.size), mask)

        image = self.transform(image)
        labels = row[self.label_cols].to_numpy(dtype=np.float32)
        return image, labels, image_id


def safe_auc(y_true, scores):
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def compute_metrics(y_true, scores, label_names):
    """Compute threshold-independent AUC metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.shape != scores.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, scores={scores.shape}")

    per_label_auc = {}
    per_label_support = {}
    aucs = []

    for column, label in enumerate(label_names):
        target = y_true[:, column]
        auc = safe_auc(target, scores[:, column])
        per_label_auc[label] = auc
        per_label_support[label] = int(target.sum())
        if auc is not None:
            aucs.append(auc)

    return {
        "per_label_auc": per_label_auc,
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "micro_auc": safe_auc(y_true.ravel(), scores.ravel()),
        "per_label_support": per_label_support,
    }


def save_npz(
    output_path,
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
):
    np.savez_compressed(
        output_path,
        image_ids=np.asarray(image_ids, dtype=object),
        label_names=np.asarray(label_names, dtype=object),
        positive_prompts=np.asarray(positive_prompts, dtype=object),
        negative_prompts=np.asarray(negative_prompts, dtype=object),
        y_true=np.asarray(y_true, dtype=np.float32),
        positive_similarities=np.asarray(positive_similarities, dtype=np.float32),
        negative_similarities=np.asarray(negative_similarities, dtype=np.float32),
        margins=np.asarray(margins, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
        temperature=np.asarray(temperature, dtype=np.float32),
    )


def evaluate_checkpoint(args, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    print(f"\n========== {checkpoint_path} ==========")
    model, tokenizer, config, device = build_model_and_tokenizer(
        config_path=args.config,
        ckpt_path=str(checkpoint_path),
        device=args.device,
    )
    temperature = get_temperature(model, args.temperature)
    print(f"[ITC] temperature={temperature:.8f}")

    dataset = VinDrDataset(
        labels_csv=args.labels_csv,
        images_root=args.images_root,
        transform=get_image_transform(config["image_res"]),
        max_images=args.max_images,
        view_type=args.view_type,
        mask_root=args.mask_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    positive_text, negative_text, positive_prompts, negative_prompts = (
        encode_bare_prompt_pairs(
            model, tokenizer, dataset.label_cols, device, args.max_text_length
        )
    )
    for label, positive, negative in zip(
        dataset.label_cols, positive_prompts, negative_prompts
    ):
        print(f"[Prompt] {label!r}: positive={positive!r}, negative={negative!r}")

    all_positive = []
    all_negative = []
    all_scores = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch_index, (images, labels, image_ids) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            image_features = get_image_embeddings(model, images)
            positive_sim = image_features @ positive_text.t()
            negative_sim = image_features @ negative_text.t()

            pair_logits = torch.stack((negative_sim, positive_sim), dim=-1)
            pair_logits = pair_logits / temperature
            positive_probability = torch.softmax(pair_logits, dim=-1)[..., 1]

            all_positive.append(positive_sim.cpu().numpy())
            all_negative.append(negative_sim.cpu().numpy())
            all_scores.append(positive_probability.cpu().numpy())
            all_labels.append(labels.numpy())
            all_ids.extend(map(str, image_ids))

            if batch_index % 10 == 0 or batch_index == len(loader):
                print(f"[Eval] processed {batch_index}/{len(loader)} batches")

    positive_similarities = np.vstack(all_positive).astype(np.float32)
    negative_similarities = np.vstack(all_negative).astype(np.float32)
    scores = np.vstack(all_scores).astype(np.float32)
    y_true = np.vstack(all_labels).astype(np.float32)
    margins = positive_similarities - negative_similarities

    metrics = compute_metrics(y_true, scores, dataset.label_cols)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint_path.stem
    prefix = f"vindr_bare_pair_itc_{args.view_type}_{stem}"
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
        "view_type": args.view_type,
        "num_images": int(len(all_ids)),
        "label_names": dataset.label_cols,
        "positive_prompts": dict(zip(dataset.label_cols, positive_prompts)),
        "negative_prompts": dict(zip(dataset.label_cols, negative_prompts)),
        "scoring": "softmax([negative_similarity, positive_similarity] / temperature)[1]",
        "uses_itm": False,
        "temperature": temperature,
        "scores_file": str(scores_path),
        "classification": metrics,
    }
    with metrics_path.open("w") as handle:
        json.dump(result, handle, indent=2)

    print(f"[Result] macro AUC: {metrics['macro_auc']}")
    print(f"[Result] saved: {scores_path}")
    print(f"[Result] saved: {metrics_path}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="ALBEF VinDr zero-shot evaluation with bare +/- ITC prompts"
    )
    parser.add_argument("--config", default="configs/Pretrain.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--output_dir", default="vindr_bare_pair_itc_results")
    parser.add_argument(
        "--view_type", choices=("original", "lung", "heart"), default="original"
    )
    parser.add_argument(
        "--mask_root",
        default=None,
        help="Required for lung/heart; masks are mask_root/id[:2]/id.png",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--max_text_length", type=int, default=32)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override model.temp. By default the learned checkpoint value is used.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.view_type != "original" and args.mask_root is None:
        raise ValueError("--mask_root is required when --view_type is lung or heart")

    results = {}
    for checkpoint in args.checkpoints:
        results[Path(checkpoint).name] = evaluate_checkpoint(args, checkpoint)

    combined_path = Path(args.output_dir) / (
        f"vindr_bare_pair_itc_{args.view_type}_all_checkpoints.json"
    )
    with combined_path.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"[Result] saved: {combined_path}")


if __name__ == "__main__":
    main()
