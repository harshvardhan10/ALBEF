# Joint-fusion model implementations

Each row links directly to the full architecture source and its experiment guide. All six remain separate implementations.

| Architecture | Full model code | Experiment guide |
|---|---|---|
| FC | [model_pretrain_multiview_fusion.py](../../src/cxr_vlm/models/model_pretrain_multiview_fusion.py) | [fc guide](fc/README.md) |
| Mean | [model_pretrain_multiview_mean_fusion.py](../../src/cxr_vlm/models/model_pretrain_multiview_mean_fusion.py) | [mean guide](mean/README.md) |
| Gated | [model_pretrain_multiview_gated_fusion.py](../../src/cxr_vlm/models/model_pretrain_multiview_gated_fusion.py) | [gated guide](gated/README.md) |
| Original-anchored residual | [model_pretrain_multiview_residual_fusion.py](../../src/cxr_vlm/models/model_pretrain_multiview_residual_fusion.py) | [residual guide](residual/README.md) |
| Two-layer cross-view Transformer | [model_pretrain_multiview_transformer_fusion.py](../../src/cxr_vlm/models/model_pretrain_multiview_transformer_fusion.py) | [transformer guide](transformer/README.md) |
| Two-layer anchored cross-attention | [model_pretrain_multiview_cross_attention_fusion.py](../../src/cxr_vlm/models/model_pretrain_multiview_cross_attention_fusion.py) | [cross_attention guide](cross_attention/README.md) |

For score-level combination of three separately trained models, see the [late-fusion guide](../late_fusion/README.md).
