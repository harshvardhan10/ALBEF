"""Original dual-AUC selection contract, extracted without changing selection rules."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

CHECKPOINT_ROLES = ("best_cardiomegaly_auc", "best_macro_auc_stable")

@dataclass
class VinDrClassificationMetrics:
    epoch: int
    view_type: str
    num_images: int
    cardiomegaly_auc: float
    macro_auc_all_evaluable: float
    macro_auc_stable: float
    macro_auc: float
    micro_auc: Optional[float]
    num_all_evaluable_auc_labels: int
    num_stable_auc_labels: int
    num_valid_auc_labels: int
    macro_auc_all_evaluable_labels: List[str]
    macro_auc_stable_labels: List[str]
    per_label_auc: Dict[str, Optional[float]]
    skipped_auc_labels: Dict[str, str]
    cardiomegaly_best_f1: float
    cardiomegaly_best_threshold: float
    cardiomegaly_score_min: float
    cardiomegaly_score_max: float
    cardiomegaly_score_mean: float
    cardiomegaly_score_std: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DualAUCSelectionState:
    best_cardiomegaly_auc: float = float("-inf")
    best_cardiomegaly_epoch: int = -1
    best_macro_auc: float = float("-inf")
    best_macro_epoch: int = -1
    validations_without_improvement: int = 0

    @classmethod
    def from_dict(
        cls,
        value: Optional[dict],
    ) -> "DualAUCSelectionState":
        if not value:
            return cls()
        return cls(
            best_cardiomegaly_auc=float(
                value.get(
                    "best_cardiomegaly_auc",
                    float("-inf"),
                )
            ),
            best_cardiomegaly_epoch=int(
                value.get(
                    "best_cardiomegaly_epoch",
                    -1,
                )
            ),
            best_macro_auc=float(
                value.get(
                    "best_macro_auc",
                    float("-inf"),
                )
            ),
            best_macro_epoch=int(
                value.get(
                    "best_macro_epoch",
                    -1,
                )
            ),
            validations_without_improvement=int(
                value.get(
                    "validations_without_improvement",
                    0,
                )
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def consider(
        self,
        metrics: VinDrClassificationMetrics,
        *,
        cardiomegaly_min_delta: float,
        macro_min_delta: float,
    ) -> Tuple[bool, bool]:
        improved_cardio = bool(
            np.isfinite(metrics.cardiomegaly_auc)
            and (
                metrics.cardiomegaly_auc
                > self.best_cardiomegaly_auc
                + float(cardiomegaly_min_delta)
            )
        )
        improved_macro = bool(
            np.isfinite(metrics.macro_auc_stable)
            and (
                metrics.macro_auc_stable
                > self.best_macro_auc
                + float(macro_min_delta)
            )
        )

        if improved_cardio:
            self.best_cardiomegaly_auc = float(
                metrics.cardiomegaly_auc
            )
            self.best_cardiomegaly_epoch = int(
                metrics.epoch
            )

        if improved_macro:
            self.best_macro_auc = float(
                metrics.macro_auc_stable
            )
            self.best_macro_epoch = int(
                metrics.epoch
            )

        if improved_cardio or improved_macro:
            self.validations_without_improvement = 0
        else:
            self.validations_without_improvement += 1

        return improved_cardio, improved_macro
