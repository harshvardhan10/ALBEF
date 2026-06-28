#!/usr/bin/env python3
"""
Check whether CXR images are anatomically horizontal or vertical.

Definitions:
  - horizontal_lungs:
      lungs are side-by-side in the current image orientation.
  - vertical_lungs_candidate:
      lung-like regions appear stacked vertically, suggesting a 90-degree rotated image.
  - canvas_portrait / canvas_landscape:
      purely based on image dimensions. This is NOT the same as anatomical orientation.

Supports:
  - PNG/JPG images
  - DICOM images, if pydicom is installed
  - CSV manifests, e.g. VinDr image_labels_test.csv
  - JSON manifests, e.g. MIMIC-CXR pretraining JSON

Install dependencies:
  pip install numpy pandas pillow opencv-python tqdm pydicom
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DICOM_EXTS = {".dcm", ".dicom"}


# -----------------------------
# Path collection
# -----------------------------

def collect_recursive(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS.union(DICOM_EXTS):
            files.append(p)
    return sorted(files)


def collect_from_csv(
    csv_path: Path,
    images_root: Path,
    id_col: Optional[str],
    path_col: Optional[str],
    filename_template: Optional[str],
) -> List[Tuple[str, Path]]:
    df = pd.read_csv(csv_path)

    if path_col is not None:
        if path_col not in df.columns:
            raise ValueError(f"path_col='{path_col}' not found. Columns: {list(df.columns)}")

        pairs = []
        for _, row in df.iterrows():
            image_id = str(row[id_col]) if id_col and id_col in df.columns else Path(str(row[path_col])).stem
            p = Path(str(row[path_col]))
            if not p.is_absolute():
                p = images_root / p
            pairs.append((image_id, p))
        return pairs

    if id_col is None:
        id_col = df.columns[0]

    if id_col not in df.columns:
        raise ValueError(f"id_col='{id_col}' not found. Columns: {list(df.columns)}")

    if filename_template is None:
        filename_template = "{image_id}.png"

    pairs = []
    for image_id in df[id_col].astype(str).tolist():
        rel = filename_template.format(image_id=image_id)
        pairs.append((image_id, images_root / rel))

    return pairs


def collect_from_json(
    json_path: Path,
    images_root: Path,
    path_keys: List[str],
    id_keys: List[str],
) -> List[Tuple[str, Path]]:
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # common patterns: {"train": [...]} or {"data": [...]}
        for key in ["data", "train", "images", "annotations"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError("JSON manifest must be a list, or a dict containing a list under data/train/images/annotations.")

    pairs = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        img_path = None
        for k in path_keys:
            if k in item and item[k]:
                img_path = str(item[k])
                break

        if img_path is None:
            continue

        image_id = None
        for k in id_keys:
            if k in item and item[k]:
                image_id = str(item[k])
                break

        p = Path(img_path)
        if not p.is_absolute():
            p = images_root / p

        if image_id is None:
            image_id = p.stem

        pairs.append((image_id, p))

    return pairs


# -----------------------------
# Image loading
# -----------------------------

def load_image_as_gray(path: Path) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    """
    Returns:
      gray image as float32, shape H x W
      metadata dict
    """
    meta = {
        "rows": None,
        "columns": None,
        "view_position": None,
        "photometric": None,
        "load_error": None,
    }

    suffix = path.suffix.lower()

    try:
        if suffix in DICOM_EXTS:
            try:
                import pydicom
            except ImportError:
                meta["load_error"] = "pydicom_not_installed"
                return None, meta

            ds = pydicom.dcmread(str(path), force=True)

            meta["rows"] = int(getattr(ds, "Rows", 0)) if getattr(ds, "Rows", None) is not None else None
            meta["columns"] = int(getattr(ds, "Columns", 0)) if getattr(ds, "Columns", None) is not None else None
            meta["view_position"] = str(getattr(ds, "ViewPosition", "")) or None
            meta["photometric"] = str(getattr(ds, "PhotometricInterpretation", "")) or None

            arr = ds.pixel_array.astype(np.float32)

            # CXR DICOM convention: MONOCHROME1 often needs inversion for display.
            if meta["photometric"] == "MONOCHROME1":
                arr = arr.max() - arr

            return arr, meta

        else:
            img = Image.open(path).convert("L")
            arr = np.asarray(img).astype(np.float32)
            meta["rows"] = arr.shape[0]
            meta["columns"] = arr.shape[1]
            return arr, meta

    except Exception as e:
        meta["load_error"] = f"{type(e).__name__}: {e}"
        return None, meta


# -----------------------------
# Orientation heuristic
# -----------------------------

def normalize_gray(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.float32)

    lo, hi = np.percentile(gray, [1, 99])
    if hi <= lo:
        lo, hi = float(gray.min()), float(gray.max())

    if hi <= lo:
        return np.zeros_like(gray, dtype=np.float32)

    gray = np.clip(gray, lo, hi)
    gray = (gray - lo) / (hi - lo + 1e-8)
    return gray


def remove_border_components(mask: np.ndarray) -> np.ndarray:
    """
    Remove connected components that touch image border.
    This helps remove black background outside the patient.
    """
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    h, w = mask.shape
    keep = np.zeros_like(mask, dtype=np.uint8)

    for comp_id in range(1, num_labels):
        x, y, bw, bh, area = stats[comp_id]
        touches_border = x <= 1 or y <= 1 or (x + bw) >= (w - 1) or (y + bh) >= (h - 1)
        if not touches_border:
            keep[labels == comp_id] = 1

    return keep


def find_two_lung_like_components(gray: np.ndarray) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Heuristic lung detector:
      - lungs are usually among darker regions inside the thorax
      - remove components touching border
      - keep plausible internal dark components
      - return two largest plausible components

    This is intentionally lightweight. It is for QC flagging, not clinical segmentation.
    """
    g = normalize_gray(gray)

    # Resize for speed and stable morphology.
    max_side = 512
    h0, w0 = g.shape
    scale = max_side / max(h0, w0) if max(h0, w0) > max_side else 1.0

    if scale != 1.0:
        g_small = cv2.resize(g, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    else:
        g_small = g.copy()

    h, w = g_small.shape

    # Ignore extreme edges.
    margin_y = int(0.03 * h)
    margin_x = int(0.03 * w)

    work = g_small.copy()

    # Dark candidate mask. Percentile threshold is more stable than absolute threshold.
    inner = work[margin_y:h - margin_y, margin_x:w - margin_x]
    dark_thr = np.percentile(inner, 42)
    dark = (work < dark_thr).astype(np.uint8)

    # Remove image border/background components.
    dark[:margin_y, :] = 0
    dark[h - margin_y:, :] = 0
    dark[:, :margin_x] = 0
    dark[:, w - margin_x:] = 0
    dark = remove_border_components(dark)

    # Morphological cleanup.
    kernel = np.ones((5, 5), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)

    comps = []
    img_area = h * w

    for comp_id in range(1, num_labels):
        x, y, bw, bh, area = stats[comp_id]
        cx, cy = centroids[comp_id]

        area_frac = area / img_area

        # Plausible lung-field component constraints.
        if area_frac < 0.005:
            continue
        if area_frac > 0.35:
            continue

        # Avoid tiny slivers and extreme edge artifacts.
        if bw < 0.08 * w or bh < 0.08 * h:
            continue
        if cx < 0.08 * w or cx > 0.92 * w:
            continue
        if cy < 0.08 * h or cy > 0.92 * h:
            continue

        comps.append({
            "area": float(area),
            "area_frac": float(area_frac),
            "x": float(x),
            "y": float(y),
            "w": float(bw),
            "h": float(bh),
            "cx": float(cx / w),
            "cy": float(cy / h),
        })

    comps = sorted(comps, key=lambda c: c["area"], reverse=True)[:2]

    debug = {
        "dark_threshold": float(dark_thr),
        "num_components": float(len(comps)),
    }

    return comps, debug


def classify_anatomical_orientation(gray: np.ndarray) -> Dict[str, object]:
    comps, debug = find_two_lung_like_components(gray)

    result = {
        "anatomy_orientation": "unknown",
        "anatomy_confidence": 0.0,
        "lung_dx_norm": None,
        "lung_dy_norm": None,
        "num_lung_components": len(comps),
        "notes": "",
    }

    if len(comps) < 2:
        result["notes"] = "Could not find two plausible lung-like components."
        return result

    c1, c2 = comps[0], comps[1]

    dx = abs(c1["cx"] - c2["cx"])
    dy = abs(c1["cy"] - c2["cy"])

    result["lung_dx_norm"] = float(dx)
    result["lung_dy_norm"] = float(dy)

    # Higher dx means lungs are side-by-side.
    # Higher dy means lungs may be stacked vertically.
    if dx > 1.35 * dy and dx > 0.18:
        result["anatomy_orientation"] = "horizontal_lungs"
        result["anatomy_confidence"] = float(min(1.0, dx / (dy + 1e-6) / 4.0))
        result["notes"] = "Two lung-like regions are separated mainly left-right."

    elif dy > 1.35 * dx and dy > 0.18:
        result["anatomy_orientation"] = "vertical_lungs_candidate"
        result["anatomy_confidence"] = float(min(1.0, dy / (dx + 1e-6) / 4.0))
        result["notes"] = "Two lung-like regions are separated mainly top-bottom; inspect manually."

    else:
        result["anatomy_orientation"] = "ambiguous"
        result["anatomy_confidence"] = 0.25
        result["notes"] = "Two lung-like components found, but separation is not clearly horizontal or vertical."

    return result


def canvas_orientation(height: int, width: int) -> str:
    if height > width:
        return "canvas_portrait"
    if width > height:
        return "canvas_landscape"
    return "canvas_square"


# -----------------------------
# Main scan
# -----------------------------

def scan_dataset(pairs: List[Tuple[str, Path]], dataset_name: str, output_csv: Path, max_images: Optional[int]) -> pd.DataFrame:
    rows = []

    if max_images is not None:
        pairs = pairs[:max_images]

    for image_id, path in tqdm(pairs, desc=f"Scanning {dataset_name}"):
        path = Path(path)

        row = {
            "dataset": dataset_name,
            "image_id": image_id,
            "path": str(path),
            "exists": path.exists(),
            "rows": None,
            "columns": None,
            "canvas_orientation": None,
            "view_position": None,
            "photometric": None,
            "candidate_sideways_by_metadata": False,
            "anatomy_orientation": "not_loaded",
            "anatomy_confidence": 0.0,
            "lung_dx_norm": None,
            "lung_dy_norm": None,
            "num_lung_components": 0,
            "load_error": None,
            "notes": "",
        }

        if not path.exists():
            row["load_error"] = "file_not_found"
            rows.append(row)
            continue

        gray, meta = load_image_as_gray(path)

        row["rows"] = meta.get("rows")
        row["columns"] = meta.get("columns")
        row["view_position"] = meta.get("view_position")
        row["photometric"] = meta.get("photometric")
        row["load_error"] = meta.get("load_error")

        if row["rows"] is not None and row["columns"] is not None:
            row["canvas_orientation"] = canvas_orientation(int(row["rows"]), int(row["columns"]))

            # Metadata-only suspicious case:
            # frontal view but landscape canvas can indicate possible sideways export.
            if str(row["view_position"]).upper() in {"PA", "AP"} and int(row["columns"]) > int(row["rows"]):
                row["candidate_sideways_by_metadata"] = True

        if gray is None:
            rows.append(row)
            continue

        anatomy = classify_anatomical_orientation(gray)
        row.update(anatomy)

        rows.append(row)

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    return df


def print_summary(df: pd.DataFrame, dataset_name: str):
    print("\n" + "=" * 80)
    print(f"Summary: {dataset_name}")
    print("=" * 80)

    print("\nCanvas orientation counts:")
    print(df["canvas_orientation"].value_counts(dropna=False))

    print("\nAnatomical orientation counts:")
    print(df["anatomy_orientation"].value_counts(dropna=False))

    print("\nMetadata sideways candidates:")
    print(df["candidate_sideways_by_metadata"].value_counts(dropna=False))

    flagged = df[
        (df["anatomy_orientation"] == "vertical_lungs_candidate")
        | (df["candidate_sideways_by_metadata"] == True)
    ]

    print(f"\nFlagged images for manual inspection: {len(flagged)}")

    if len(flagged) > 0:
        cols = [
            "image_id",
            "path",
            "rows",
            "columns",
            "canvas_orientation",
            "view_position",
            "anatomy_orientation",
            "anatomy_confidence",
            "lung_dx_norm",
            "lung_dy_norm",
            "notes",
        ]
        print(flagged[cols].head(20).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, required=True, help="Example: MIMIC-CXR or VinDr-CXR")
    parser.add_argument("--images_root", type=str, required=True)

    # Choose one manifest mode, or use recursive scan.
    parser.add_argument("--manifest_csv", type=str, default=None)
    parser.add_argument("--manifest_json", type=str, default=None)
    parser.add_argument("--recursive", action="store_true")

    # CSV options.
    parser.add_argument("--id_col", type=str, default=None)
    parser.add_argument("--path_col", type=str, default=None)
    parser.add_argument(
        "--filename_template",
        type=str,
        default="{image_id}.png",
        help='Used with CSV id_col. Example: "{image_id}.png" or "train/{image_id}.dicom"',
    )

    # JSON options.
    parser.add_argument(
        "--json_path_keys",
        type=str,
        default="image,image_path,path,jpg_path,dicom_path",
        help="Comma-separated candidate keys for image path in JSON manifest.",
    )
    parser.add_argument(
        "--json_id_keys",
        type=str,
        default="image_id,dicom_id,id",
        help="Comma-separated candidate keys for image id in JSON manifest.",
    )

    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--max_images", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    images_root = Path(args.images_root)
    output_csv = Path(args.output_csv)

    if args.manifest_csv:
        pairs = collect_from_csv(
            csv_path=Path(args.manifest_csv),
            images_root=images_root,
            id_col=args.id_col,
            path_col=args.path_col,
            filename_template=args.filename_template,
        )

    elif args.manifest_json:
        pairs = collect_from_json(
            json_path=Path(args.manifest_json),
            images_root=images_root,
            path_keys=[x.strip() for x in args.json_path_keys.split(",") if x.strip()],
            id_keys=[x.strip() for x in args.json_id_keys.split(",") if x.strip()],
        )

    elif args.recursive:
        files = collect_recursive(images_root)
        pairs = [(p.stem, p) for p in files]

    else:
        raise ValueError("Use one of: --manifest_csv, --manifest_json, or --recursive")

    print(f"[Input] Dataset: {args.dataset_name}")
    print(f"[Input] Number of candidate files: {len(pairs)}")
    print(f"[Output] CSV: {output_csv}")

    df = scan_dataset(
        pairs=pairs,
        dataset_name=args.dataset_name,
        output_csv=output_csv,
        max_images=args.max_images,
    )

    print_summary(df, args.dataset_name)


if __name__ == "__main__":
    main()