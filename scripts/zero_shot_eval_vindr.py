"""

Zero-shot evaluation of ALBEF checkpoints on VinDr-CXR (classification).

What it does:
- Loads one or more ALBEF checkpoints (pretrained on MIMIC).
- Builds multi-prompt text embeddings for each VinDr label (via src.py).
- Runs zero-shot classification on the VinDr test set (image-level labels).
- Computes:
    - Per-label ROC–AUC
    - Macro and micro ROC–AUC
    - Per-label F1 (classification, global threshold)
    - Macro and micro F1
    - mAP@10 (mean average precision at 10, multilabel classification)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader

from src import (
    build_model_and_tokenizer,
    get_image_transform,
    get_label_text_embeddings,
    get_image_embeddings,
)


# ==========================
# Dataset
# ==========================

class VinDrClassificationDataset(Dataset):
    """
    Simple image-level classification dataset for VinDr-CXR.
    Expects:
      - CSV with columns: image_id, <label_1>, <label_2>, ...
      - PNG images at: images_root / f"{image_id}.png"
    """

    def __init__(self, csv_path, images_root, transform=None, max_images=None):
        self.df = pd.read_csv(csv_path)
        self.images_root = Path(images_root)
        self.transform = transform

        # assume first column is image_id, rest are label columns
        self.id_col = self.df.columns[0]
        self.label_cols = list(self.df.columns[1:])

        # filter to rows with existing PNGs
        self.df["__has_png__"] = self.df[self.id_col].apply(
            lambda x: (self.images_root / f"{x}.png").exists()
        )
        self.df = self.df[self.df["__has_png__"]].reset_index(drop=True)
        if max_images is not None:
            self.df = self.df.iloc[:max_images].reset_index(drop=True)

        self.num_labels = len(self.label_cols)
        print(
            f"[VinDrClassificationDataset] Using {len(self.df)} images, "
            f"{self.num_labels} labels"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row[self.id_col]
        img_path = self.images_root / f"{image_id}.png"
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # labels: 0/1 per pathology
        labels = row[self.label_cols].values.astype(np.float32)
        return img, labels, image_id


# ==========================
# Metrics
# ==========================

def compute_map_at_k(y_true, scores, k=10):
    """
    Compute mAP@k for multilabel classification.

    For each sample:
      - Rank labels by score (descending).
      - Consider top-k labels.
      - Compute "AP@k" as the average of precisions at ranks where the label is positive,
        normalized by min(#positives, k).
    Then average AP@k over all samples that have at least one positive label.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    N, L = y_true.shape
    ap_list = []

    for i in range(N):
        y = y_true[i]
        s = scores[i]
        pos_idx = np.where(y == 1)[0]
        if len(pos_idx) == 0:
            continue

        order = np.argsort(-s)      # descending
        topk = order[:k]

        hits = 0
        precisions = []
        for rank, idx in enumerate(topk, start=1):
            if y[idx] == 1:
                hits += 1
                precisions.append(hits / rank)

        if len(precisions) == 0:
            ap = 0.0
        else:
            denom = min(len(pos_idx), k)
            ap = float(np.sum(precisions) / denom)
        ap_list.append(ap)

    if len(ap_list) == 0:
        return None
    return float(np.mean(ap_list))


def compute_classification_metrics(
        y_true,
        scores,
        label_names,
        threshold=0.5
):
    """
    y_true:  (N, L) 0/1
    scores:  (N, L) continuous similarities (higher = more positive)

    Computes:
      - per-label ROC–AUC + macro/micro AUC
      - per-label F1 + macro/micro F1 (using a global threshold)
      - mAP@10 (multilabel classification)
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    N, L = y_true.shape
    assert scores.shape == (N, L)

    metrics = {}

    # ----- ROC–AUC -----
    per_label_auc = {}
    auc_values = []
    for j, label in enumerate(label_names):
        y = y_true[:, j]
        # need both classes present
        if len(np.unique(y)) < 2:
            per_label_auc[label] = None
            continue
        try:
            auc = roc_auc_score(y, scores[:, j])
            per_label_auc[label] = float(auc)
            auc_values.append(auc)
        except ValueError:
            per_label_auc[label] = None

    metrics["per_label_auc"] = per_label_auc
    metrics["macro_auc"] = float(np.mean(auc_values)) if len(auc_values) > 0 else None

    # micro-AUC: flatten all labels
    try:
        metrics["micro_auc"] = float(roc_auc_score(y_true.ravel(), scores.ravel()))
    except ValueError:
        metrics["micro_auc"] = None

    # ----- F1 (using a global threshold) -----
    y_pred = (scores >= threshold).astype(int)

    per_label_f1 = {}
    f1_values = []
    for j, label in enumerate(label_names):
        y = y_true[:, j]
        y_hat = y_pred[:, j]
        if len(np.unique(y)) < 2:
            per_label_f1[label] = None
            continue
        f1 = f1_score(y, y_hat)
        per_label_f1[label] = float(f1)
        f1_values.append(f1)
    metrics["per_label_f1"] = per_label_f1
    metrics["macro_f1"] = float(np.mean(f1_values)) if len(f1_values) > 0 else None

    # micro-F1 (flatten)
    metrics["micro_f1"] = float(f1_score(y_true.ravel(), y_pred.ravel()))

    # ----- mAP@10 (multilabel classification) -----
    metrics["map_at_10"] = compute_map_at_k(y_true, scores, k=10)

    return metrics


# ==========================
# Main evaluation logic
# ==========================

def evaluate_checkpoint(
    ckpt_path,
    config_path,
    csv_path,
    images_root,
    batch_size,
    num_workers,
    device_str,
    output_dir,
    max_images=None,
    threshold=0.5,
):
    """
    Runs zero-shot classification evaluation for a single checkpoint.
    Returns a metrics dict.
    """
    print(f"\n========== Evaluating checkpoint: {ckpt_path} ==========")
    ckpt_path = Path(ckpt_path)
    ckpt_name = ckpt_path.stem

    # Build model + tokenizer + config from shared src.py helper
    model, tokenizer, config, device = build_model_and_tokenizer(
        config_path=config_path,
        ckpt_path=str(ckpt_path),
        device=device_str,
    )

    # Dataset & DataLoader
    image_res = config["image_res"]
    test_transform = get_image_transform(image_res)

    dataset = VinDrClassificationDataset(
        csv_path=csv_path,
        images_root=images_root,
        transform=test_transform,
        max_images=max_images,
    )
    label_names = dataset.label_cols

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Text embeddings (multi-prompt per label; shared code with localization)
    label_embs = get_label_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        labels=label_names,
        device=device,
        max_length=32,
    )
    label_embs = label_embs.to(device)  # (L, D)
    print("[Text] Label embeddings shape:", label_embs.shape)

    # Inference loop
    all_scores = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for i, (images, labels, image_ids) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)    # (B,3,H,W)
            labels_np = labels.numpy().astype(np.float32)     # (B,L)

            image_embs = get_image_embeddings(model, images)  # (B,D)
            sims = image_embs @ label_embs.t()                # (B, L)
            scores = sims.cpu().numpy()

            all_scores.append(scores)
            all_labels.append(labels_np)
            all_ids.extend(list(image_ids))

            if (i + 1) % 10 == 0 or (i + 1) == len(data_loader):
                print(f"[Eval] Processed {i+1}/{len(data_loader)} batches")

    all_scores = np.vstack(all_scores)   # (N, L)
    all_labels = np.vstack(all_labels)   # (N, L)

    print("Scores shape:", all_scores.shape)
    print("Labels shape:", all_labels.shape)

    # Classification metrics
    cls_metrics = compute_classification_metrics(
        y_true=all_labels,
        scores=all_scores,
        label_names=label_names,
        threshold=threshold,
    )

    results = {
        "checkpoint": str(ckpt_path),
        "num_images": int(all_scores.shape[0]),
        "label_names": list(label_names),
        "classification": cls_metrics,
        "threshold": float(threshold),
    }

    # Save JSON
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"vindr_zero_shot_{ckpt_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Output] Saved metrics to: {out_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot VinDr-CXR evaluation for ALBEF checkpoints"
    )

    parser.add_argument("--config", type=str, default="configs/Pretrain.yaml")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="List of checkpoint paths to evaluate")
    parser.add_argument("--labels_csv", type=str, required=True,
                        help="Path to image-level labels CSV (e.g. image_labels_test.csv)")
    parser.add_argument("--images_root", type=str, required=True,
                        help="Root folder containing VinDr test PNGs")
    parser.add_argument("--output_dir", type=str, default="vindr_zero_shot_results")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Global threshold used for F1 computation")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Optional: limit number of images for quick debugging")

    return parser.parse_args()


def main():
    args = parse_args()

    all_results = {}
    for ckpt in args.checkpoints:
        res = evaluate_checkpoint(
            ckpt_path=ckpt,
            config_path=args.config,
            csv_path=args.labels_csv,
            images_root=args.images_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_str=args.device,
            output_dir=args.output_dir,
            max_images=args.max_images,
            threshold=args.threshold,
        )
        all_results[Path(ckpt).name] = res

    combined_path = Path(args.output_dir) / "vindr_zero_shot_all_checkpoints.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Output] Saved combined metrics to: {combined_path}")


if __name__ == "__main__":
    main()
