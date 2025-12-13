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
        image_id = str(row[self.id_col])
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


def compute_classification_metrics(y_true, scores, label_names, threshold=0.5):
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

    # micro-AUC
    try:
        metrics["micro_auc"] = float(roc_auc_score(y_true.ravel(), scores.ravel()))
    except ValueError:
        metrics["micro_auc"] = None

    # ----- F1 (global threshold) -----
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
    metrics["micro_f1"] = float(f1_score(y_true.ravel(), y_pred.ravel()))

    # ----- mAP@10 -----
    metrics["map_at_10"] = compute_map_at_k(y_true, scores, k=10)

    return metrics


# ==========================
# Saving helpers
# ==========================

def save_scores_npz(out_path: Path,
                    image_ids: list,
                    label_names: list,
                    scores: np.ndarray,
                    y_true: np.ndarray = None):
    """
    Compact, fast load for downstream:
      - image_ids (N,)
      - label_names (L,)
      - scores (N,L)
      - y_true (optional) (N,L)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_ids": np.asarray(image_ids, dtype=object),
        "label_names": np.asarray(label_names, dtype=object),
        "scores": np.asarray(scores, dtype=np.float32),
    }
    if y_true is not None:
        payload["y_true"] = np.asarray(y_true, dtype=np.float32)
    np.savez_compressed(out_path, **payload)
    print(f"[Output] Saved per-image scores (npz): {out_path}")


def save_scores_csv(out_path: Path,
                    image_ids: list,
                    label_names: list,
                    scores: np.ndarray,
                    y_true: np.ndarray = None):
    """
    Human-readable wide CSV:
      columns: image_id, score::<label_1>, ..., score::<label_L>
      optionally also y::<label> columns.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(scores, columns=[f"score::{lb}" for lb in label_names])
    df.insert(0, "image_id", image_ids)

    if y_true is not None:
        ydf = pd.DataFrame(y_true, columns=[f"y::{lb}" for lb in label_names])
        df = pd.concat([df, ydf], axis=1)

    df.to_csv(out_path, index=False)
    print(f"[Output] Saved per-image scores (csv): {out_path}")


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
    save_scores: bool = True,
    save_scores_csv_flag: bool = False,
):
    print(f"\n========== Evaluating checkpoint: {ckpt_path} ==========")
    ckpt_path = Path(ckpt_path)
    ckpt_name = ckpt_path.stem

    model, tokenizer, config, device = build_model_and_tokenizer(
        config_path=config_path,
        ckpt_path=str(ckpt_path),
        device=device_str,
    )

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

    label_embs = get_label_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        labels=label_names,
        device=device,
        max_length=32,
    )
    label_embs = label_embs.to(device)  # (L, D)
    print("[Text] Label embeddings shape:", tuple(label_embs.shape))

    all_scores = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for i, (images, labels, image_ids) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            labels_np = labels.numpy().astype(np.float32)

            image_embs = get_image_embeddings(model, images)  # (B,D) normalized
            sims = image_embs @ label_embs.t()                # (B, L)
            scores = sims.cpu().numpy().astype(np.float32)

            all_scores.append(scores)
            all_labels.append(labels_np)
            all_ids.extend(list(map(str, image_ids)))

            if (i + 1) % 10 == 0 or (i + 1) == len(data_loader):
                print(f"[Eval] Processed {i+1}/{len(data_loader)} batches")

    all_scores = np.vstack(all_scores)   # (N, L)
    all_labels = np.vstack(all_labels)   # (N, L)

    print("[Eval] Scores shape:", all_scores.shape)
    print("[Eval] Labels shape:", all_labels.shape)
    print("[Eval] Num image_ids:", len(all_ids))

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
        "scores_file_npz": None,
        "scores_file_csv": None,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Save per-image per-label scores ----
    if save_scores:
        scores_npz_path = output_dir / f"vindr_zero_shot_scores_{ckpt_name}.npz"
        save_scores_npz(
            out_path=scores_npz_path,
            image_ids=all_ids,
            label_names=label_names,
            scores=all_scores,
            y_true=all_labels,
        )
        results["scores_file_npz"] = str(scores_npz_path)

        if save_scores_csv_flag:
            scores_csv_path = output_dir / f"vindr_zero_shot_scores_{ckpt_name}.csv"
            save_scores_csv(
                out_path=scores_csv_path,
                image_ids=all_ids,
                label_names=label_names,
                scores=all_scores,
                y_true=all_labels,
            )
            results["scores_file_csv"] = str(scores_csv_path)

    # ---- Save JSON metrics ----
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
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Global threshold used for F1 computation")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Optional: limit number of images for quick debugging")

    parser.add_argument("--save_scores", action="store_true",
                        help="Save per-image per-label scores as .npz (recommended)")
    parser.add_argument("--save_scores_csv", action="store_true",
                        help="Also save per-image per-label scores as wide CSV (optional)")

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
            save_scores=args.save_scores,
            save_scores_csv_flag=args.save_scores_csv,
        )
        all_results[Path(ckpt).name] = res

    combined_path = Path(args.output_dir) / "vindr_zero_shot_all_checkpoints.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Output] Saved combined metrics to: {combined_path}")


if __name__ == "__main__":
    main()
