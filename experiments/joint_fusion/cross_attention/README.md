# Two-layer anchored cross-attention fusion

**Open the full model code:** [model_pretrain_multiview_cross_attention_fusion.py](../../../src/cxr_vlm/models/model_pretrain_multiview_cross_attention_fusion.py). 

The original token supplies the query; lung and heart tokens supply auxiliary memory. Two cross-attention layers update the original anchor.

## Code map

| Component | Source |
|---|---|
| Fusion operator | [ViewCrossAttentionFusion](../../../src/cxr_vlm/models/model_pretrain_multiview_cross_attention_fusion.py#L136) |
| Token fusion | [ALBEF._fuse_tokens](../../../src/cxr_vlm/models/model_pretrain_multiview_cross_attention_fusion.py#L430) |
| Three-view image encoding | [ALBEF.encode_image_views](../../../src/cxr_vlm/models/model_pretrain_multiview_cross_attention_fusion.py#L479) |
| Training losses | [ALBEF.forward](../../../src/cxr_vlm/models/model_pretrain_multiview_cross_attention_fusion.py#L559) |
| Momentum update | [ALBEF._momentum_update](../../../src/cxr_vlm/models/model_pretrain_multiview_cross_attention_fusion.py#L880) |
| Training entry point | [trainer](Pretrain_A0_multiview_cross_attention_fusion_dual_auc.py) |
| Historical configuration | [Pretrain_A0_multiview_cross_attention_fusion_dual_auc.yaml](../../../configs/historical/original_configs_new/Pretrain_A0_multiview_cross_attention_fusion_dual_auc.yaml) |
| ITC-margin branch attribution | [extractor](localization/extract_cross_attention_fusion_itc_margin_gradcam_heatmaps.py) |

## Experiment identity

Three aligned original/lung/heart image encoders produce one fused token sequence for the ALBEF objectives. 

The attribution extractor aggregates positive unnormalized branch maps, then applies one maximum normalization. 

Use [the registered launcher](../../../tools/run_experiment.py) from the repository root. 
