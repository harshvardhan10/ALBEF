#!/usr/bin/env python3
"""Extract BioViL-T phrase-grounding maps for the exact ALBEF-selected cases.

The ALBEF visualization notebook saves one row per selected image in:

    selected_50_margin_cases.csv

This script reads those exact image IDs and generates BOTH BioViL-T pathology
maps for every image:

    image 1 -> Cardiomegaly
            -> Pleural effusion
    image 2 -> Cardiomegaly
            -> Pleural effusion
    ...

Thus, for 50 ALBEF-selected images, 100 BioViL-T maps are generated.

No new image sampling is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image


DEFAULT_PROMPTS = {
    "Cardiomegaly": "cardiomegaly",
    "Pleural effusion": "pleural effusion",
}


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels_csv",
        required=True,
        help="VinDr image-level labels CSV.",
    )

    parser.add_argument(
        "--images_root",
        required=True,
        help="Directory containing VinDr image PNGs.",
    )

    parser.add_argument(
        "--albef_selection_csv",
        required=True,
        help=(
            "selected_50_margin_cases.csv saved by the ALBEF "
            "visualization notebook."
        ),
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where BioViL-T heatmaps will be saved.",
    )

    parser.add_argument(
        "--target_labels",
        nargs="+",
        default=["Cardiomegaly", "Pleural effusion"],
    )

    parser.add_argument(
        "--prompts_json",
        default=None,
        help="Optional JSON mapping target labels to BioViL-T query phrases.",
    )

    parser.add_argument(
        "--expected_images",
        type=int,
        default=50,
        help="Expected number of unique ALBEF-selected images.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--interpolation",
        choices=["nearest", "bilinear", "bicubic"],
        default="bilinear",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Prompt handling
# ============================================================

def load_prompts(args: argparse.Namespace) -> dict[str, str]:
    prompts = dict(DEFAULT_PROMPTS)

    if args.prompts_json:
        supplied = json.loads(args.prompts_json)

        if not isinstance(supplied, dict):
            raise ValueError(
                "--prompts_json must decode to a JSON object"
            )

        prompts.update(
            {str(key): str(value) for key, value in supplied.items()}
        )

    missing = [
        label
        for label in args.target_labels
        if label not in prompts
    ]

    if missing:
        raise ValueError(
            "No prompt supplied for: "
            + ", ".join(missing)
        )

    return {
        label: prompts[label]
        for label in args.target_labels
    }


# ============================================================
# CSV helpers
# ============================================================

def find_id_column(frame: pd.DataFrame) -> str:
    for candidate in (
        "image_id",
        "imageId",
        "imageid",
        "image_name",
        "id",
    ):
        if candidate in frame.columns:
            return candidate

    return str(frame.columns[0])


def build_exact_albef_selection(
    args: argparse.Namespace,
    labels_frame: pd.DataFrame,
    labels_id_column: str,
    prompts: dict[str, str],
) -> pd.DataFrame:
    """Create 2 image-label rows per ALBEF-selected image.

    No random sampling occurs here.

    Example:
        ALBEF selected CSV:
            img_A
            img_B

        becomes:
            img_A | Cardiomegaly
            img_A | Pleural effusion
            img_B | Cardiomegaly
            img_B | Pleural effusion
    """

    selection_path = Path(args.albef_selection_csv)

    if not selection_path.is_file():
        raise FileNotFoundError(
            f"ALBEF selection CSV not found: {selection_path}"
        )

    albef_selected = pd.read_csv(selection_path)

    if albef_selected.empty:
        raise ValueError(
            "ALBEF selection CSV is empty"
        )

    albef_id_column = find_id_column(albef_selected)

    albef_selected = albef_selected.rename(
        columns={albef_id_column: "image_id"}
    )

    albef_selected["image_id"] = (
        albef_selected["image_id"].astype(str)
    )

    # There should be one row per selected CXR.
    if albef_selected["image_id"].duplicated().any():
        duplicates = (
            albef_selected.loc[
                albef_selected["image_id"].duplicated(
                    keep=False
                ),
                "image_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "ALBEF selection contains duplicate image IDs: "
            f"{duplicates[:10]}"
        )

    if (
        args.expected_images is not None
        and len(albef_selected) != args.expected_images
    ):
        raise ValueError(
            f"Expected {args.expected_images} ALBEF-selected images "
            f"but found {len(albef_selected)}"
        )

    # --------------------------------------------------------
    # Validate labels CSV
    # --------------------------------------------------------

    labels_frame = labels_frame.copy()

    labels_frame[labels_id_column] = (
        labels_frame[labels_id_column].astype(str)
    )

    if labels_frame[labels_id_column].duplicated().any():
        raise ValueError(
            "Labels CSV contains duplicate image IDs"
        )

    for label in args.target_labels:
        if label not in labels_frame.columns:
            raise KeyError(
                f"{label!r} is absent from labels CSV. "
                f"Available columns: {list(labels_frame.columns)}"
            )

        labels_frame[label] = pd.to_numeric(
            labels_frame[label],
            errors="raise",
        )

        if not labels_frame[label].isin([0, 1]).all():
            raise ValueError(
                f"{label!r} must contain binary 0/1 labels"
            )

    labels_lookup = labels_frame.set_index(
        labels_id_column
    )

    selected_ids = albef_selected["image_id"].tolist()

    missing_ids = [
        image_id
        for image_id in selected_ids
        if image_id not in labels_lookup.index
    ]

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} ALBEF-selected images are absent "
            f"from the labels CSV. First entries: {missing_ids[:10]}"
        )

    # --------------------------------------------------------
    # Expand every selected image to every target label
    # --------------------------------------------------------

    records: list[dict[str, Any]] = []

    for albef_order, (_, row) in enumerate(
        albef_selected.iterrows(),
        start=1,
    ):
        image_id = str(row["image_id"])

        for label in args.target_labels:
            gt = int(
                labels_lookup.loc[
                    image_id,
                    label,
                ]
            )

            # If the ALBEF selection CSV itself contains the GT
            # columns, make sure the two sources agree.
            if label in albef_selected.columns:
                albef_gt = int(row[label])

                if albef_gt != gt:
                    raise ValueError(
                        f"{image_id} / {label}: "
                        f"ALBEF selection GT={albef_gt}, "
                        f"labels CSV GT={gt}"
                    )

            record = {
                "image_id": image_id,
                "label": label,
                "prompt": prompts[label],
                "ground_truth": gt,
                "albef_order": albef_order,
            }

            if "stratum" in albef_selected.columns:
                record["stratum"] = str(
                    row["stratum"]
                )

            records.append(record)

    selected = pd.DataFrame(records)

    if selected.duplicated(
        ["image_id", "label"]
    ).any():
        raise RuntimeError(
            "Duplicate image-label pairs were created"
        )

    expected_pairs = (
        len(albef_selected)
        * len(args.target_labels)
    )

    if len(selected) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} image-label pairs "
            f"but created {len(selected)}"
        )

    return selected


# ============================================================
# Image handling
# ============================================================

def resolve_image_path(
    images_root: Path,
    image_id: str,
) -> Path:

    direct = images_root / f"{image_id}.png"

    if direct.is_file():
        return direct

    for suffix in (".jpg", ".jpeg"):
        candidate = images_root / f"{image_id}{suffix}"

        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No image found for {image_id} under {images_root}"
    )


# ============================================================
# BioViL-T map handling
# ============================================================

def as_2d_float_tensor(
    value,
) -> torch.Tensor:
    """Convert BioViL-T map to aligned 256x256 tensor.

    BioViL-T's valid image region is [16:240, 16:240].

    The 16-pixel border is explicitly zeroed rather than
    extrapolated.
    """

    if isinstance(value, torch.Tensor):
        returned = (
            value
            .detach()
            .float()
            .cpu()
            .squeeze()
        )
    else:
        returned = torch.as_tensor(
            value,
            dtype=torch.float32,
        ).squeeze()

    if returned.shape != (256, 256):
        raise ValueError(
            "Expected BioViL-T map shape "
            f"(256, 256), got {tuple(returned.shape)}"
        )

    valid_crop = returned[16:240, 16:240]

    if not torch.isfinite(valid_crop).all():
        raise ValueError(
            "Central BioViL-T map contains "
            "NaN or infinity"
        )

    aligned = torch.zeros(
        (256, 256),
        dtype=torch.float32,
    )

    aligned[16:240, 16:240] = valid_crop

    return aligned


def normalize_valid_region(
    raw: torch.Tensor,
) -> torch.Tensor:
    """Min-max normalize only the valid 224x224 region."""

    central = raw[16:240, 16:240]

    low = central.min()
    high = central.max()

    normalized = torch.zeros_like(raw)

    if float(high - low) > 1e-8:
        normalized[16:240, 16:240] = (
            central - low
        ) / (
            high - low
        )

    return normalized


def safe_torch_load(
    path: Path,
):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def load_existing_raw_map(
    saved: dict,
    expected_image_id: str,
    expected_label: str,
) -> torch.Tensor:
    """Support both old and current BioViL-T payload key names."""

    if str(saved.get("image_id")) != expected_image_id:
        raise ValueError(
            f"Existing heatmap image mismatch: "
            f"{saved.get('image_id')} != {expected_image_id}"
        )

    if str(saved.get("label")) != expected_label:
        raise ValueError(
            f"Existing heatmap label mismatch: "
            f"{saved.get('label')} != {expected_label}"
        )

    # Your earlier scripts/notebooks used both names at
    # different points. Accept either for compatibility.
    if "similarity_map_raw_aligned" in saved:
        value = saved["similarity_map_raw_aligned"]

    elif "similarity_map_raw" in saved:
        value = saved["similarity_map_raw"]

    else:
        raise KeyError(
            "Existing BioViL-T file contains neither "
            "'similarity_map_raw_aligned' nor "
            "'similarity_map_raw'"
        )

    return as_2d_float_tensor(value)


# ============================================================
# BioViL-T model
# ============================================================

def load_biovil_t(
    device: torch.device,
):
    try:
        from health_multimodal.image.utils import (
            get_image_inference,
            ImageModelType,
        )

        from health_multimodal.text.utils import (
            get_bert_inference,
            BertEncoderType,
        )

        from health_multimodal.vlp.inference_engine import (
            ImageTextInferenceEngine,
        )

    except ImportError as error:
        raise RuntimeError(
            "BioViL-T dependencies are missing. "
            "Install with:\n"
            "pip install --upgrade hi-ml-multimodal"
        ) from error

    text_inference = get_bert_inference(
        BertEncoderType.BIOVIL_T_BERT
    )

    image_inference = get_image_inference(
        ImageModelType.BIOVIL_T
    )

    engine = ImageTextInferenceEngine(
        image_inference_engine=image_inference,
        text_inference_engine=text_inference,
    )

    engine.to(device)

    return engine


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is not available"
        )

    device = torch.device(args.device)

    labels_csv = Path(args.labels_csv)
    images_root = Path(args.images_root)
    output_dir = Path(args.output_dir)
    maps_dir = output_dir / "maps"

    if not labels_csv.is_file():
        raise FileNotFoundError(labels_csv)

    if not images_root.is_dir():
        raise FileNotFoundError(images_root)

    maps_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    labels_frame = pd.read_csv(
        labels_csv
    )

    labels_id_column = find_id_column(
        labels_frame
    )

    labels_frame[labels_id_column] = (
        labels_frame[labels_id_column]
        .astype(str)
    )

    prompts = load_prompts(args)

    # --------------------------------------------------------
    # Exact ALBEF selection
    # --------------------------------------------------------

    selected = build_exact_albef_selection(
        args=args,
        labels_frame=labels_frame,
        labels_id_column=labels_id_column,
        prompts=prompts,
    )

    selection_path = (
        output_dir
        / "selected_albef_cases_expanded.csv"
    )

    selected.to_csv(
        selection_path,
        index=False,
    )

    num_unique_images = (
        selected["image_id"].nunique()
    )

    print()
    print(
        "[Selection] Exact ALBEF-selected cases"
    )
    print(
        f"[Selection] unique images = "
        f"{num_unique_images}"
    )
    print(
        f"[Selection] target labels = "
        f"{len(args.target_labels)}"
    )
    print(
        f"[Selection] total BioViL-T maps = "
        f"{len(selected)}"
    )
    print(
        f"[Selection] source = "
        f"{args.albef_selection_csv}"
    )
    print(
        f"[Selection] expanded CSV = "
        f"{selection_path}"
    )

    print()
    print(
        selected.groupby(
            ["label", "ground_truth"]
        ).size().to_string()
    )
    print()

    # --------------------------------------------------------
    # Load BioViL-T once
    # --------------------------------------------------------

    engine = load_biovil_t(
        device
    )

    manifest_records: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # Generate maps
    # --------------------------------------------------------

    for position, row in enumerate(
        selected.itertuples(index=False),
        start=1,
    ):
        image_id = str(row.image_id)
        label = str(row.label)
        prompt = str(row.prompt)
        ground_truth = int(row.ground_truth)
        albef_order = int(row.albef_order)

        safe_label = (
            label
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
        )

        output_path = (
            maps_dir
            / f"{image_id}__{safe_label}.pt"
        )

        image_path = resolve_image_path(
            images_root,
            image_id,
        )

        # ----------------------------------------------------
        # Existing map
        # ----------------------------------------------------

        if (
            output_path.exists()
            and not args.overwrite
        ):
            print(
                f"[{position:03d}/"
                f"{len(selected):03d}] "
                f"skip {output_path.name}"
            )

            saved = safe_torch_load(
                output_path
            )

            raw = load_existing_raw_map(
                saved=saved,
                expected_image_id=image_id,
                expected_label=label,
            )

            normalized = normalize_valid_region(
                raw
            )

        # ----------------------------------------------------
        # New BioViL-T map
        # ----------------------------------------------------

        else:
            with torch.inference_mode():
                value = (
                    engine
                    .get_similarity_map_from_raw_data(
                        image_path=image_path,
                        query_text=prompt,
                        interpolation=args.interpolation,
                    )
                )

            raw = as_2d_float_tensor(
                value
            )

            normalized = normalize_valid_region(
                raw
            )

            central_raw = raw[
                16:240,
                16:240,
            ]

            with Image.open(
                image_path
            ) as image:
                original_size = tuple(
                    int(x)
                    for x in image.size
                )

            payload = {
                "image_id": image_id,
                "image_path": str(image_path),
                "original_size_wh": original_size,

                "label": label,
                "ground_truth": ground_truth,
                "prompt": prompt,

                "albef_selection_order": albef_order,
                "albef_selection_csv": str(
                    args.albef_selection_csv
                ),

                "model_name": "BioViL-T",
                "model_type": "biovil_t",

                "method": (
                    "native_patch_text_cosine_similarity"
                ),

                "interpolation": (
                    args.interpolation
                ),

                # Keep BOTH names so your previous
                # scripts/notebooks remain compatible.
                "similarity_map_raw": raw,
                "similarity_map_raw_aligned": raw,

                "similarity_map_vis": normalized,

                "valid_region": [
                    16,
                    240,
                    16,
                    240,
                ],

                "raw_min": float(
                    central_raw.min()
                ),

                "raw_max": float(
                    central_raw.max()
                ),

                "raw_mean": float(
                    central_raw.mean()
                ),

                "raw_std": float(
                    central_raw.std(
                        unbiased=False
                    )
                ),
            }

            if hasattr(row, "stratum"):
                payload["albef_stratum"] = str(
                    row.stratum
                )

            torch.save(
                payload,
                output_path,
            )

            print(
                f"[{position:03d}/"
                f"{len(selected):03d}] "
                f"saved {output_path.name}"
            )

        # ----------------------------------------------------
        # Manifest
        # ----------------------------------------------------

        central_raw = raw[
            16:240,
            16:240,
        ]

        record = {
            "image_id": image_id,
            "albef_order": albef_order,

            "label": label,
            "prompt": prompt,
            "ground_truth": ground_truth,

            "image_path": str(
                image_path
            ),

            "heatmap_path": str(
                output_path
            ),

            "map_height": int(
                raw.shape[0]
            ),

            "map_width": int(
                raw.shape[1]
            ),

            "valid_y0": 16,
            "valid_y1": 240,
            "valid_x0": 16,
            "valid_x1": 240,

            "raw_min": float(
                central_raw.min()
            ),

            "raw_max": float(
                central_raw.max()
            ),

            "raw_mean": float(
                central_raw.mean()
            ),

            "raw_std": float(
                central_raw.std(
                    unbiased=False
                )
            ),
        }

        if hasattr(row, "stratum"):
            record["stratum"] = str(
                row.stratum
            )

        manifest_records.append(
            record
        )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest = pd.DataFrame(
        manifest_records
    )

    manifest_path = (
        output_dir
        / "manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final consistency checks
    # --------------------------------------------------------

    expected_pairs = (
        num_unique_images
        * len(args.target_labels)
    )

    if len(manifest) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} maps, "
            f"but manifest contains "
            f"{len(manifest)}"
        )

    pair_counts = (
        manifest
        .groupby("image_id")["label"]
        .nunique()
    )

    bad_images = pair_counts[
        pair_counts
        != len(args.target_labels)
    ]

    if not bad_images.empty:
        raise RuntimeError(
            "Some images do not have all target "
            f"labels:\n{bad_images}"
        )

    print()
    print("=" * 70)
    print("[Done]")
    print(
        f"Unique ALBEF images : "
        f"{num_unique_images}"
    )
    print(
        f"Labels per image     : "
        f"{len(args.target_labels)}"
    )
    print(
        f"BioViL-T maps        : "
        f"{len(manifest)}"
    )
    print(
        f"Manifest             : "
        f"{manifest_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()