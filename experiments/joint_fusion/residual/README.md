# Original-anchored residual fusion

**Open the full model code:** [model_pretrain_multiview_residual_fusion.py](../../../src/cxr_vlm/models/model_pretrain_multiview_residual_fusion.py). This is the complete implementation, not an import facade.

The original-view representation is the anchor; learned sigmoid gates control lung and heart residual contributions.

## Code map

| Component | Source |
|---|---|
| Fusion operator | [OriginalAnchoredResidualFusion](../../../src/cxr_vlm/models/model_pretrain_multiview_residual_fusion.py#L21) |
| Token fusion | [ALBEF._fuse_tokens](../../../src/cxr_vlm/models/model_pretrain_multiview_residual_fusion.py#L259) |
| Three-view image encoding | [ALBEF.encode_image_views](../../../src/cxr_vlm/models/model_pretrain_multiview_residual_fusion.py#L308) |
| Training losses | [ALBEF.forward](../../../src/cxr_vlm/models/model_pretrain_multiview_residual_fusion.py#L388) |
| Momentum update | [ALBEF._momentum_update](../../../src/cxr_vlm/models/model_pretrain_multiview_residual_fusion.py#L709) |
| Training entry point | [trainer](Pretrain_A0_multiview_residual_fusion_dual_auc.py) |
| Historical configuration | [Pretrain_A0_multiview_residual_fusion_dual_auc.yaml](../../../configs/historical/original_configs_new/Pretrain_A0_multiview_residual_fusion_dual_auc.yaml) |
| ITC-margin branch attribution | [extractor](localization/extract_residual_fusion_itc_margin_gradcam_heatmaps.py) |

## Experiment identity

Three aligned original/lung/heart image encoders produce one fused token sequence for the ALBEF objectives. Keep this architecture separate from the other five fusion implementations; similar trainer structure is not permission to merge them.

The attribution extractor aggregates positive unnormalized branch maps, then applies one maximum normalization. It is not a fusion-layer attention matrix or a late score ensemble. Final run checkpoints are unavailable here; the linked code/config mapping is not a claim of numerical reproduction.

Use [the registered launcher](../../../tools/run_experiment.py) from the repository root. The generic-named multi-view classification evaluator currently constructs cross-attention ALBEF only; it must not be assumed valid for every fusion family.
