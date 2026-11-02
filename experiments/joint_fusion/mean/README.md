# Mean fusion

**Open the full model code:** [model_pretrain_multiview_mean_fusion.py](../../../src/cxr_vlm/models/model_pretrain_multiview_mean_fusion.py).
The fusion operator averages the three aligned token streams. The operator has no learned fusion parameters; the complete ALBEF model still has trainable encoders.

## Code map

| Component | Source |
|---|---|
| Fusion operator | [SimpleMeanFusion](../../../src/cxr_vlm/models/model_pretrain_multiview_mean_fusion.py#L21) |
| Token fusion | [ALBEF._fuse_tokens](../../../src/cxr_vlm/models/model_pretrain_multiview_mean_fusion.py#L184) |
| Three-view image encoding | [ALBEF.encode_image_views](../../../src/cxr_vlm/models/model_pretrain_multiview_mean_fusion.py#L233) |
| Training losses | [ALBEF.forward](../../../src/cxr_vlm/models/model_pretrain_multiview_mean_fusion.py#L313) |
| Momentum update | [ALBEF._momentum_update](../../../src/cxr_vlm/models/model_pretrain_multiview_mean_fusion.py#L634) |
| Training entry point | [trainer](Pretrain_A0_multiview_mean_fusion_dual_auc.py) |
| Historical configuration | [Pretrain_A0_multiview_mean_fusion_dual_auc.yaml](../../../configs/historical/original_configs_new/Pretrain_A0_multiview_mean_fusion_dual_auc.yaml) |
| ITC-margin branch attribution | [extractor](localization/extract_mean_fusion_itc_margin_gradcam_heatmaps.py) |

## Experiment identity

Three aligned original/lung/heart image encoders produce one fused token sequence for the ALBEF objectives.

The attribution extractor aggregates positive unnormalized branch maps, then applies one maximum normalization. 
Use [the registered launcher](../../../tools/run_experiment.py) from the repository root.
