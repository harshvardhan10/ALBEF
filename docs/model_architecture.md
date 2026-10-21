# Architecture review map

Each joint model processes aligned original, lung and heart views through separate vision encoders, with a shared text/projection pathway and an online/momentum pair. Review the source rather than treating the variants as interchangeable.

| Fusion | Model source | Distinguishing behavior |
|---|---|---|
| FC | [model_pretrain_multiview_fusion.py](../models/model_pretrain_multiview_fusion.py) | Concatenate corresponding view tokens, project and normalize; equal-average initialization |
| Mean | [model_pretrain_multiview_mean_fusion.py](../models/model_pretrain_multiview_mean_fusion.py) | Tokenwise arithmetic mean |
| Gated | [model_pretrain_multiview_gated_fusion.py](../models/model_pretrain_multiview_gated_fusion.py) | Sample-specific weights shared across tokens |
| Residual | [model_pretrain_multiview_residual_fusion.py](../models/model_pretrain_multiview_residual_fusion.py) | Original representation plus gated anatomical contributions |
| Transformer | [model_pretrain_multiview_transformer_fusion.py](../models/model_pretrain_multiview_transformer_fusion.py) | Two-layer cross-view token-position processing followed by mean pooling |
| Cross-attention | [model_pretrain_multiview_cross_attention_fusion.py](../models/model_pretrain_multiview_cross_attention_fusion.py) | Original token queries fixed lung/heart memory through two anchored layers; no mean pooling |

Cross-attention adds view embeddings once. At each aligned position, query length is 1 and memory length is 2; the second layer reuses the same anatomical memory. Online and momentum fusion modules are distinct. Preserve state prefixes view_fusion.* and view_fusion_m.*.

Patch-head refinement is a separate workflow, not a fusion variant. Existing duplicated model/trainer bodies are still present intentionally until checkpoint and baseline heatmap compatibility can be tested. No shared-model extraction has yet been claimed complete.
