# Experiment guide

For the shortest route to each full architecture and method implementation, start with the [model-code index](model_code.md).


| Stage | Thesis sections | Entry points / analysis |
|---|---|---|
| Baselines | 5.3, 6.2 | [Original training](../experiments/baseline/Pretrain.py), [bare-prompt evaluator](../experiments/classification/bare_prompt/zero_shot_eval_vindr_pos_neg_bare_prompts.py), [ITC-margin maps](../experiments/localization/itc/extract_itc_margin_gradcam_heatmaps.py) |
| Direct guidance | 5.4, 6.3 | [A0–A3 trainer](../experiments/anatomy_guidance/Pretrain_anatomy_prior.py), [A4 trainer](../experiments/anatomy_guidance/Pretrain_anatomy_prior_A4.py) |
| Anatomical views | 5.5, 6.4 | [Single-view dual-AUC trainer](../experiments/single_view/Pretrain_A0_multiview_dual_auc.py), [mask-cache trainer](../experiments/single_view/Pretrain_A0_maskcache.py), [online-mask trainer](../experiments/single_view/Pretrain_A0_chexmask.py), [precomputed-view trainer](../experiments/single_view/Pretrain_A0_precomputed.py) |
| Score combination | 5.6.1, 6.5.1 | [Normalized ensembles](../experiments/late_fusion/ensemble_zero_shot_scores.py), [fitted ensemble weights](../experiments/late_fusion/learn_dual_checkpoint_ensemble_weights.py) |
| Joint fusion | 5.6.2, 6.5.2–6.5.4 | Six trainer/model/extractor families below |
| Evaluation | 5.7 | [Validation-thresholded F1](../experiments/evaluation/evaluate_zero_shot_f1_from_validation.py), [pointing-game notebook](../notebooks/evaluation/localization_pointing_game.ipynb) |

## Guidance run identity

A1, A2, A3, A3.1, A3.2 and A4 each require separate Cardiomegaly and Pleural Effusion run records. A0 is the original-image reference.

| Variant | Configuration evidence |
|---|---|---|
| A0 | [A0 YAML](../configs/historical/original_configs/Pretrain_anatomy_prior_A0.yaml) |
| A1 | [A1 YAML](../configs/historical/original_configs/Pretrain_anatomy_prior_A1.yaml) | 
| A2 | [A2 YAML](../configs/historical/original_configs/Pretrain_anatomy_prior_A2.yaml) |
| A3 | [Current A3 YAML](../configs/historical/original_configs/Pretrain_anatomy_prior_A3.yaml) |
| A3.1 | Historical commit `ea61e1e` | 
| A3.2 | Historical commit `c9a5803` |
| A4 | [A4 YAML](../configs/historical/original_configs/Pretrain_anatomy_prior_A4.yaml) |

## Fusion families

| Family | Trainer | Config |
|---|---|---|
| FC | [Trainer](../experiments/joint_fusion/fc/Pretrain_A0_multiview_fusion_dual_auc.py) | [Config](../configs/historical/original_configs_new/Pretrain_A0_multiview_fusion_dual_auc.yaml) |
| Mean | [Trainer](../experiments/joint_fusion/mean/Pretrain_A0_multiview_mean_fusion_dual_auc.py) | [Config](../configs/historical/original_configs_new/Pretrain_A0_multiview_mean_fusion_dual_auc.yaml) |
| Gated | [Trainer](../experiments/joint_fusion/gated/Pretrain_A0_multiview_gated_fusion_dual_auc.py) | [Config](../configs/historical/original_configs_new/Pretrain_A0_multiview_gated_fusion_dual_auc.yaml) |
| Residual | [Trainer](../experiments/joint_fusion/residual/Pretrain_A0_multiview_residual_fusion_dual_auc.py) | [Config](../configs/historical/original_configs_new/Pretrain_A0_multiview_residual_fusion_dual_auc.yaml) |
| Transformer | [Trainer](../experiments/joint_fusion/transformer/Pretrain_A0_multiview_transformer_fusion_dual_auc.py) | [Config](../configs/historical/original_configs_new/Pretrain_A0_multiview_transformer2_mean_dual_auc.yaml) |
| Cross-attention | [Trainer](../experiments/joint_fusion/cross_attention/Pretrain_A0_multiview_cross_attention_fusion_dual_auc.py) | [Config](../configs/historical/original_configs_new/Pretrain_A0_multiview_cross_attention_fusion_dual_auc.yaml) |

Each family has a separately named `scripts_new/extract_<family>_fusion_itc_margin_gradcam_heatmaps.py`. Their model definitions are linked in [methodology](methodology.md).

The file [zero_shot_eval_vindr_multiview_pos_neg_bare_prompts.py](../experiments/classification/bare_prompt/zero_shot_eval_vindr_multiview_pos_neg_bare_prompts.py) can be modified the import the required fusion model (currently imports the cross-attention model) It is the common dispatcher for all six fusion architectures.

## External references and historical work

- BioViL-T: [native extraction](../experiments/external_baselines/biovil_t/extract_biovil_t_phrase_grounding.py) and [same-ALBEF-case extraction](../experiments/external_baselines/biovil_t/extract_biovil_t_same_albef_cases.py).
- CheXzero: [comparison notebook](../notebooks/comparisons/visualize_chexzero_checkpoints_and_model_comparison.ipynb) 
- VQA, NLVR, retrieval, visual entailment and RefCOCO: inherited ALBEF functionality; preserve source and attribution.
