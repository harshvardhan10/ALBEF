# Localization methods

A heatmap is meaningful only with its score target and attention source.

| Entry point | Method |
|---|---|
| scripts_new/extract_albef_itc_gradcam_heatmaps.py | ALBEF ITC score Grad-CAM |
| scripts_new/extract_itc_margin_gradcam_heatmaps.py | Single-view positive-minus-negative ITC Grad-CAM |
| scripts/extract_crossattn_gradcam_heatmaps.py | Single-view ALBEF text-image cross-attention Grad-CAM; not the fusion model |
| scripts_new/extract_*_fusion_itc_margin_gradcam_heatmaps.py | Fusion-specific ITC-margin ViT self-attention Grad-CAM |
| scripts/extract_patch_head_heatmaps.py | Learned spatial patch head |
| scripts_new/extract_biovil_t_phrase_grounding.py | BioViL-T native phrase grounding |
| baselines/chexzero/extract_chexzero_multi_checkpoint_attention_gradcam.py | CheXzero attention Grad-CAM |

The new cross-attention fusion extractor backpropagates `(positive_similarity - negative_similarity) / temperature` through the actual fusion model into each visual branch. Default `--vit_layer -2` is the second-to-last block (index 10 for 12 blocks). CLS-to-patch attention times its gradient is rectified per head before head averaging. Each positive map is divided by its maximum, or remains zero. The composite sums raw positive branch maps and normalizes once. Upsampling uses bilinear interpolation with align_corners=False. Signed raw maps are also saved for comparison.

Prompts are the bare label and `no <label>`; No finding uses Finding as its negative. No ITM score or direct fusion-attention weights are substituted. This method is consistent with the other controlled fusion extractors, not a new claim of anatomical localization quality. FROC implementations remain exploratory.
