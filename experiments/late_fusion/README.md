# Late score fusion

Late fusion is score combination, not a separate neural-network architecture. Its inputs come from three [single-view ALBEF models](../../src/cxr_vlm/models/model_pretrain.py) (original/lung/heart).

| Method | Actual implementation |
|---|---|
| Validation-normalized equal-weight ensemble | [ensemble_zero_shot_scores.py](ensemble_zero_shot_scores.py) |
| Validation-fitted simplex weights | [learn_dual_checkpoint_ensemble_weights.py](learn_dual_checkpoint_ensemble_weights.py) |

The code aligns image IDs and label names before combination and fits normalization/calibration on validation scores.
