# Two-layer cross-view Transformer fusion

**Open the full model code:** [model_pretrain_multiview_transformer_fusion.py](../../../src/cxr_vlm/models/model_pretrain_multiview_transformer_fusion.py). 

At each aligned token position, two Transformer layers mix the three view tokens, followed by mean pooling.

## Code map

| Component | Source |
|---|---|
| Fusion operator | [ViewTransformerFusion](../../../src/cxr_vlm/models/model_pretrain_multiview_transformer_fusion.py#L35) |
| Token fusion | [ALBEF._fuse_tokens](../../../src/cxr_vlm/models/model_pretrain_multiview_transformer_fusion.py#L313) |
| Three-view image encoding | [ALBEF.encode_image_views](../../../src/cxr_vlm/models/model_pretrain_multiview_transformer_fusion.py#L362) |
| Training losses | [ALBEF.forward](../../../src/cxr_vlm/models/model_pretrain_multiview_transformer_fusion.py#L442) |
| Momentum update | [ALBEF._momentum_update](../../../src/cxr_vlm/models/model_pretrain_multiview_transformer_fusion.py#L763) |
| Training entry point | [trainer](Pretrain_A0_multiview_transformer_fusion_dual_auc.py) |
| Historical configuration | [Pretrain_A0_multiview_transformer2_mean_dual_auc.yaml](../../../configs/historical/original_configs_new/Pretrain_A0_multiview_transformer2_mean_dual_auc.yaml) |
| ITC-margin branch attribution | [extractor](localization/extract_transformer_fusion_itc_margin_gradcam_heatmaps.py) |

## Experiment identity

Three aligned original/lung/heart image encoders produce one fused token sequence for the ALBEF objectives. 

The attribution extractor aggregates positive unnormalized branch maps, then applies one maximum normalization. 

Use [the registered launcher](../../../tools/run_experiment.py) from the repository root. 
