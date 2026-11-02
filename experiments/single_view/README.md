# Original, lung-only and heart-only ALBEF

All three views use the [same full ALBEF architecture](../../src/cxr_vlm/models/model_pretrain.py). The image input changes; the code does not introduce a new architecture for each mask.

| View | Model code | Historical dual-AUC config |
|---|---|---|
| original | [ALBEF](../../src/cxr_vlm/models/model_pretrain.py) | [config](../../configs/historical/original_configs/Pretrain_A0_original_dual_auc.yaml) |
| lung | [ALBEF](../../src/cxr_vlm/models/model_pretrain.py) | [config](../../configs/historical/original_configs/Pretrain_A0_lung_dual_auc.yaml) |
| heart | [ALBEF](../../src/cxr_vlm/models/model_pretrain.py) | [config](../../configs/historical/original_configs/Pretrain_A0_heart_dual_auc.yaml) |

## View construction and training

- [Single-view dual-AUC trainer](Pretrain_A0_multiview_dual_auc.py)
- [Cached-mask dataset](../../src/cxr_vlm/datasets/chexmask_cached_mask_dataset.py) and [online mask dataset](../../src/cxr_vlm/datasets/chexmask_view_dataset.py) — preserve these distinct input-generation paths.
- [Mask-cache trainer](Pretrain_A0_maskcache.py), [online-mask trainer](Pretrain_A0_chexmask.py) and [precomputed-view trainer](Pretrain_A0_precomputed.py) — historical alternatives, not proven interchangeable runs.
