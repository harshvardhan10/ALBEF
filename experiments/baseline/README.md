# Baseline single-view ALBEF

**Full model implementation:** [model_pretrain.py](../../src/cxr_vlm/models/model_pretrain.py). This is the actual architecture/objective code, not a legacy compatibility loader.

| Read this | Purpose |
|---|---|
| [ALBEF.__init__](../../src/cxr_vlm/models/model_pretrain.py#L21) | Vision/text encoders, projections, ITM head, momentum copies and queues |
| [ALBEF.forward](../../src/cxr_vlm/models/model_pretrain.py#L88) | ITC/ITA, ITM and MLM losses |
| [ViT](../../src/cxr_vlm/models/vit.py) | Image encoder and attention |
| [XBERT](../../src/cxr_vlm/models/xbert.py) | Text and image-conditioned BERT |
| [Historical baseline trainer](Pretrain.py) | Training/checkpoint orchestration |

For anatomical-view training, continue to the [original/lung/heart guide](../single_view/README.md). For A0–A4 guidance, use the [guidance guide](../anatomy_guidance/README.md).
