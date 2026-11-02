# Model code: start here

The links below open the **full implementation files**. They deliberately avoid historical compatibility loaders and do not invent separate architectures where experiments share a model.

## Main thesis experiments

| Experiment | Full model implementation | Where its method differs | Guide |
|---|---|---|---|
| Baseline A0 | [ALBEF](../src/cxr_vlm/models/model_pretrain.py) | Baseline objectives and training | [Baseline](../experiments/baseline/README.md) |
| A1 guidance | [Shared ALBEF](../src/cxr_vlm/models/model_pretrain.py) | [Guidance/preservation trainer](../experiments/anatomy_guidance/Pretrain_anatomy_prior.py) | [A0–A4 method map](../experiments/anatomy_guidance/README.md) |
| A2 guidance | [Shared ALBEF](../src/cxr_vlm/models/model_pretrain.py) | [Guidance/preservation trainer](../experiments/anatomy_guidance/Pretrain_anatomy_prior.py) | [A0–A4 method map](../experiments/anatomy_guidance/README.md) |
| A3 guidance | [Shared ALBEF](../src/cxr_vlm/models/model_pretrain.py) | [Guidance/preservation trainer](../experiments/anatomy_guidance/Pretrain_anatomy_prior.py) | [A0–A4 method map](../experiments/anatomy_guidance/README.md) |
| A3.1 guidance | [Shared ALBEF](../src/cxr_vlm/models/model_pretrain.py) | [Guidance/preservation trainer](../experiments/anatomy_guidance/Pretrain_anatomy_prior.py) | [A0–A4 method map](../experiments/anatomy_guidance/README.md) |
| A3.2 guidance | [Shared ALBEF](../src/cxr_vlm/models/model_pretrain.py) | [Guidance/preservation trainer](../experiments/anatomy_guidance/Pretrain_anatomy_prior.py) | [A0–A4 method map](../experiments/anatomy_guidance/README.md) |
| A4 guidance | [Shared ALBEF](../src/cxr_vlm/models/model_pretrain.py) | [Guidance/preservation trainer](../experiments/anatomy_guidance/Pretrain_anatomy_prior_A4.py) | [A0–A4 method map](../experiments/anatomy_guidance/README.md) |
| Original / lung / heart views | [Shared ALBEF](../src/cxr_vlm/models/model_pretrain.py) | [View dataset/training](../experiments/single_view/Pretrain_A0_multiview_dual_auc.py) | [Single views](../experiments/single_view/README.md) |
| FC fusion | [model_pretrain_multiview_fusion.py](../src/cxr_vlm/models/model_pretrain_multiview_fusion.py) | Architecture-specific token fusion | [Guide](../experiments/joint_fusion/fc/README.md) |
| Mean fusion | [model_pretrain_multiview_mean_fusion.py](../src/cxr_vlm/models/model_pretrain_multiview_mean_fusion.py) | Architecture-specific token fusion | [Guide](../experiments/joint_fusion/mean/README.md) |
| Gated fusion | [model_pretrain_multiview_gated_fusion.py](../src/cxr_vlm/models/model_pretrain_multiview_gated_fusion.py) | Architecture-specific token fusion | [Guide](../experiments/joint_fusion/gated/README.md) |
| Original-anchored residual fusion | [model_pretrain_multiview_residual_fusion.py](../src/cxr_vlm/models/model_pretrain_multiview_residual_fusion.py) | Architecture-specific token fusion | [Guide](../experiments/joint_fusion/residual/README.md) |
| Two-layer cross-view Transformer fusion | [model_pretrain_multiview_transformer_fusion.py](../src/cxr_vlm/models/model_pretrain_multiview_transformer_fusion.py) | Architecture-specific token fusion | [Guide](../experiments/joint_fusion/transformer/README.md) |
| Two-layer anchored cross-attention fusion | [model_pretrain_multiview_cross_attention_fusion.py](../src/cxr_vlm/models/model_pretrain_multiview_cross_attention_fusion.py) | Architecture-specific token fusion | [Guide](../experiments/joint_fusion/cross_attention/README.md) |
| Equal/tuned late fusion | Three [single-view ALBEF models](../src/cxr_vlm/models/model_pretrain.py) | [Score fitting/combination](../experiments/late_fusion/learn_dual_checkpoint_ensemble_weights.py) | [Late fusion](../experiments/late_fusion/README.md) |

## Shared building blocks

- [Vision Transformer](../src/cxr_vlm/models/vit.py).
- [XBERT text/image-conditioned encoder](../src/cxr_vlm/models/xbert.py) and [BERT tokenizer](../src/cxr_vlm/models/tokenization_bert.py).
- [Anatomy attention extraction](../src/cxr_vlm/anatomy/attention_extract.py), [outside-support loss](../src/cxr_vlm/anatomy/losses.py) and [target token selection](../src/cxr_vlm/anatomy/token_utils.py).

A0–A4, including A3.1/A3.2, share ALBEF's architecture. The guidance losses live in the trainer and anatomy helpers; coefficient variants remain distinct experiment identities. Guidance is target-specific for Cardiomegaly and Pleural Effusion. Do not duplicate one architecture into misleading new A1/A2 model classes.

## External references and historical work

BioViL-T's [native extraction integration](../experiments/external_baselines/biovil_t/extract_biovil_t_phrase_grounding.py) is present; the full pretrained external model architecture is supplied by its external package, not implemented here. CheXzero's [comparison notebook](../notebooks/comparisons/visualize_chexzero_checkpoints_and_model_comparison.ipynb) consumes external outputs; its complete model implementation is absent from this repository. No missing implementation or result is invented.

A5/A6 remain preserved in `archive/non_thesis_experiments/` outside the main-thesis guide. The root `Pretrain*.py`, old `models/`, `scripts/` and `scripts_new/` paths are compatibility loaders; use the full source links above for reading model/method code.

Final run configurations, checkpoints and evaluation outputs are unavailable in this environment. That limits numerical reproduction, not access to the model implementations. Known source/thesis differences remain documented in [the discrepancy record](discrepancies.md).

See the [validation record](model_code_validation.md) for navigation checks, source preservation and the limits of runtime validation.
