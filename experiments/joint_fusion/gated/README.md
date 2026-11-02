# Gated fusion

**Open the full model code:** [model_pretrain_multiview_gated_fusion.py](../../../src/cxr_vlm/models/model_pretrain_multiview_gated_fusion.py).

A CLS-conditioned network predicts three study-level softmax weights, applied to corresponding tokens before normalization.

## Code map

| Component | Source |
|---|---|
| Fusion operator | [ImageLevelGatedFusion](../../../src/cxr_vlm/models/model_pretrain_multiview_gated_fusion.py#L21) |
| Token fusion | [ALBEF._fuse_tokens](../../../src/cxr_vlm/models/model_pretrain_multiview_gated_fusion.py#L242) |
| Three-view image encoding | [ALBEF.encode_image_views](../../../src/cxr_vlm/models/model_pretrain_multiview_gated_fusion.py#L291) |
| Training losses | [ALBEF.forward](../../../src/cxr_vlm/models/model_pretrain_multiview_gated_fusion.py#L371) |
| Momentum update | [ALBEF._momentum_update](../../../src/cxr_vlm/models/model_pretrain_multiview_gated_fusion.py#L692) |
| Training entry point | [trainer](Pretrain_A0_multiview_gated_fusion_dual_auc.py) |
| Historical configuration | [Pretrain_A0_multiview_gated_fusion_dual_auc.yaml](../../../configs/historical/original_configs_new/Pretrain_A0_multiview_gated_fusion_dual_auc.yaml) |
| ITC-margin branch attribution | [extractor](localization/extract_gated_fusion_itc_margin_gradcam_heatmaps.py) |

## Experiment identity

Three aligned original/lung/heart image encoders produce one fused token sequence for the ALBEF objectives. 
The attribution extractor aggregates positive unnormalized branch maps, then applies one maximum normalization. 
Use [the registered launcher](../../../tools/run_experiment.py) from the repository root.
