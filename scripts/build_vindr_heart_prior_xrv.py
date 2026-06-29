import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchxrayvision as xrv


class VinDrImageDataset(Dataset):
    """
    Minimal VinDr PNG dataset.

    Expects either:
      1) --index_csv with image_id in the first column, or
      2) no CSV, in which case all images_root/*.<image_ext> are used.

    Images are loaded as grayscale, normalized with TorchXRayVision's expected
    CXR normalization, center-cropped, and resized to 512 for ChestX-Det PSPNet.
    """

    def __init__(
        self,
        images_root: Path,
        index_csv: Optional[Path] = None,
        image_ext: str = "png",
        max_images: Optional[int] = None,
    ):
        self.images_root = Path(images_root)
        self.image_ext = image_ext.lstrip(".")

        if index_csv is not None:
            df = pd.read_csv(index_csv)
            id_col = df.columns[0]

            raw_image_ids = df[id_col].astype(str).tolist()
            image_ids = list(dict.fromkeys(raw_image_ids))  # stable de-duplication

            print(
                f"[Dataset] index_csv rows={len(raw_image_ids)} "
                f"unique_image_ids={len(image_ids)} using id_col='{id_col}'",
                flush=True,
            )
        else:
            paths = sorted(self.images_root.glob(f"*.{self.image_ext}"))
            image_ids = [p.stem for p in paths]

        records = []
        seen = set()
        missing = 0

        for image_id in image_ids:
            if image_id in seen:
                continue
            seen.add(image_id)

            p = self.images_root / f"{image_id}.{self.image_ext}"
            if p.exists():
                records.append((image_id, p))
            else:
                missing += 1

        if missing > 0:
            print(f"[Dataset] Warning: {missing} unique image_ids had no matching PNG", flush=True)

        if max_images is not None:
            records = records[: int(max_images)]

        if len(records) == 0:
            raise RuntimeError(f"No .{self.image_ext} images found under {self.images_root}")

        self.records = records
        self.transforms = [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(512),
        ]

        print(f"[Dataset] Using {len(self.records)} images from {self.images_root}", flush=True)

    def __len__(self):
        return len(self.records)

    def _load_xrv_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("L")
        arr = np.asarray(img).astype(np.float32)

        # VinDr PNGs are usually uint8 after conversion, but this also handles uint16-like PNGs.
        maxval = 255.0 if arr.max() <= 255.0 else 65535.0
        arr = xrv.datasets.normalize(arr, maxval=maxval)

        # XRV expects C,H,W.
        arr = arr[None, :, :]
        for transform in self.transforms:
            arr = transform(arr)
        arr = np.ascontiguousarray(arr).astype(np.float32)
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        image_id, path = self.records[idx]
        x = self._load_xrv_image(path)
        return x, image_id


def save_preview_png(mask: np.ndarray, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    m = np.asarray(mask, dtype=np.float32)
    m = m - float(m.min())
    denom = float(m.max()) if float(m.max()) > 1e-8 else 1.0
    m = m / denom
    img = Image.fromarray((m * 255.0).clip(0, 255).astype(np.uint8))
    img.save(out_png)


def build_heart_prior(
    images_root: Path,
    output_npy: Path,
    index_csv: Optional[Path],
    image_ext: str,
    batch_size: int,
    num_workers: int,
    device: str,
    heart_threshold: float,
    output_size: int,
    max_images: Optional[int],
    cache_dir: Optional[str],
    save_individual_masks_dir: Optional[Path],
):
    dataset = VinDrImageDataset(
        images_root=images_root,
        index_csv=index_csv,
        image_ext=image_ext,
        max_images=max_images,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    print("[Model] Loading TorchXRayVision ChestX-Det PSPNet", flush=True)
    try:
        seg_model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=cache_dir)
    except TypeError:
        seg_model = xrv.baseline_models.chestx_det.PSPNet()
    seg_model = seg_model.to(device)
    seg_model.eval()

    if not hasattr(seg_model, "targets"):
        raise RuntimeError("PSPNet model has no .targets attribute; check torchxrayvision version.")

    print(f"[Model] Targets: {seg_model.targets}", flush=True)
    heart_idx = seg_model.targets.index("Heart")
    print(f"[Model] Heart channel index = {heart_idx}", flush=True)

    sum_mask = torch.zeros((output_size, output_size), dtype=torch.float64)
    n_images = 0
    diagnostics = []

    if save_individual_masks_dir is not None:
        save_individual_masks_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for x, image_ids in tqdm(loader, desc="Segmenting VinDr train images"):
            x = x.to(device, non_blocking=True)

            out = seg_model(x)  # expected [B, 14, 512, 512]
            heart = out[:, heart_idx : heart_idx + 1, :, :]

            # Some versions/models return probabilities, others can return logits.
            # This keeps the script robust.
            if float(heart.min()) < 0.0 or float(heart.max()) > 1.0:
                heart_prob = torch.sigmoid(heart)
            else:
                heart_prob = heart.clamp(0.0, 1.0)

            heart_bin = (heart_prob >= float(heart_threshold)).float()

            heart_256 = F.interpolate(
                heart_bin,
                size=(output_size, output_size),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)

            sum_mask += heart_256[:, 0].double().cpu().sum(dim=0)
            n_images += heart_256.shape[0]

            batch_area = heart_256[:, 0].mean(dim=(1, 2)).detach().cpu().numpy()
            for img_id, area_frac in zip(image_ids, batch_area):
                diagnostics.append({
                    "image_id": str(img_id),
                    "heart_area_fraction_256": float(area_frac),
                })

            if save_individual_masks_dir is not None:
                for j, img_id in enumerate(image_ids):
                    m = heart_256[j, 0].detach().cpu().numpy().astype(np.float32)
                    np.save(save_individual_masks_dir / f"{img_id}_heart_mask_{output_size}.npy", m)

    if n_images == 0:
        raise RuntimeError("No images were processed.")

    prior = (sum_mask / float(n_images)).float().numpy()
    prior = np.clip(prior, 0.0, 1.0).astype(np.float32)

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, prior)
    print(f"[Output] Saved heart prior: {output_npy}", flush=True)
    print(
        f"[Prior] shape={prior.shape} min={prior.min():.4f} "
        f"max={prior.max():.4f} mean={prior.mean():.4f}",
        flush=True,
    )

    preview_png = output_npy.with_suffix(".png")
    save_preview_png(prior, preview_png)
    print(f"[Output] Saved preview PNG: {preview_png}", flush=True)

    diag_csv = output_npy.with_name(output_npy.stem + "_diagnostics.csv")
    pd.DataFrame(diagnostics).to_csv(diag_csv, index=False)
    print(f"[Output] Saved diagnostics: {diag_csv}", flush=True)

    meta = {
        "images_root": str(images_root),
        "index_csv": str(index_csv) if index_csv is not None else None,
        "image_ext": image_ext,
        "num_images": int(n_images),
        "model": "torchxrayvision.baseline_models.chestx_det.PSPNet",
        "target": "Heart",
        "heart_channel_index": int(heart_idx),
        "heart_threshold": float(heart_threshold),
        "output_size": int(output_size),
        "output_npy": str(output_npy),
        "preview_png": str(preview_png),
        "prior_min": float(prior.min()),
        "prior_max": float(prior.max()),
        "prior_mean": float(prior.mean()),
    }
    meta_json = output_npy.with_name(output_npy.stem + "_metadata.json")
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Output] Saved metadata: {meta_json}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a VinDr-train heart segmentation-derived anatomy prior using TorchXRayVision."
    )
    parser.add_argument("--images_root", type=str, required=True,
                        help="Directory containing VinDr train PNG images named <image_id>.png")
    parser.add_argument("--index_csv", type=str, default=None,
                        help="CSV whose first column is image_id, e.g. image_labels_train.csv. If omitted, all PNGs in images_root are used.")
    parser.add_argument("--output_npy", type=str, required=True,
                        help="Output .npy path, e.g. anatomy_priors/vindr_train_heart_prior_256.npy")
    parser.add_argument("--image_ext", type=str, default="png")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--heart_threshold", type=float, default=0.5,
                        help="Threshold applied to each predicted heart probability map before averaging.")
    parser.add_argument("--output_size", type=int, default=256,
                        help="Size of final prior. Keep 256 for your ALBEF image_res=256.")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Optional TorchXRayVision model cache directory.")
    parser.add_argument("--save_individual_masks_dir", type=str, default=None,
                        help="Optional directory to save per-image 256x256 heart masks. Usually leave unset.")
    return parser.parse_args()


def main():
    args = parse_args()
    build_heart_prior(
        images_root=Path(args.images_root),
        output_npy=Path(args.output_npy),
        index_csv=Path(args.index_csv) if args.index_csv else None,
        image_ext=args.image_ext,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        heart_threshold=args.heart_threshold,
        output_size=args.output_size,
        max_images=args.max_images,
        cache_dir=args.cache_dir,
        save_individual_masks_dir=Path(args.save_individual_masks_dir) if args.save_individual_masks_dir else None,
    )


if __name__ == "__main__":
    main()
