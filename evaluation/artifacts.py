"""Atomic result writers shared by evaluation and training."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
from evaluation.checkpoint_roles import VinDrClassificationMetrics

def atomic_torch_save(
    payload: dict,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)


def atomic_save_scores_npz(
    output_path: str | Path,
    *,
    outputs: dict,
    metrics: VinDrClassificationMetrics,
    metadata: dict = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    with open(temporary_path, "wb") as handle:
        np.savez_compressed(
            handle,
            image_ids=np.asarray(
                outputs["image_ids"],
                dtype=object,
            ),
            label_names=np.asarray(
                outputs["label_names"],
                dtype=object,
            ),
            y_true=np.asarray(
                outputs["y_true"],
                dtype=np.int8,
            ),
            scores=np.asarray(
                outputs["scores"],
                dtype=np.float32,
            ),
            epoch=np.asarray(
                metrics.epoch,
                dtype=np.int64,
            ),
            view_type=np.asarray(metrics.view_type),
            cardiomegaly_auc=np.asarray(
                metrics.cardiomegaly_auc,
                dtype=np.float32,
            ),
            macro_auc_all_evaluable=np.asarray(
                metrics.macro_auc_all_evaluable,
                dtype=np.float32,
            ),
            macro_auc_stable=np.asarray(
                metrics.macro_auc_stable,
                dtype=np.float32,
            ),
            macro_auc=np.asarray(
                metrics.macro_auc,
                dtype=np.float32,
            ),
            cardiomegaly_best_threshold=np.asarray(
                metrics.cardiomegaly_best_threshold,
                dtype=np.float32,
            ),
        )

    os.replace(temporary_path, output_path)
    from evaluation.experiment_metadata import write_score_metadata
    write_score_metadata(output_path, outputs, metrics, metadata)
