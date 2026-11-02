# Experiment guide

The [machine-readable registry](audit/experiment_registry.csv) contains 27 experiment/analysis records and candidate implementation paths. A candidate mapping is not proof of the executed run. [Historical lineage](audit/historical_lineage.csv) records exact evidence commits; [discrepancies](discrepancies.md) distinguishes unresolved claims.

## Thesis stages

| Stage | Thesis sections | Entry points / analysis |
|---|---|---|
| Baselines | 5.3, 6.2 | [Original training](../Pretrain.py), [bare-prompt evaluator](../scripts_new/zero_shot_eval_vindr_pos_neg_bare_prompts.py), [ITC-margin maps](../scripts_new/extract_itc_margin_gradcam_heatmaps.py) |
| Direct guidance | 5.4, 6.3 | [A0–A3 trainer](../Pretrain_anatomy_prior.py), [A4 trainer](../Pretrain_anatomy_prior_A4.py) |
| Anatomical views | 5.5, 6.4 | [Single-view dual-AUC trainer](../Pretrain_A0_multiview_dual_auc.py), [mask-cache trainer](../Pretrain_A0_maskcache.py), [online-mask trainer](../Pretrain_A0_chexmask.py), [precomputed-view trainer](../Pretrain_A0_precomputed.py) |
| Score combination | 5.6.1, 6.5.1 | [Normalized ensembles](../scripts/ensemble_zero_shot_scores.py), [fitted ensemble weights](../scripts_new/learn_dual_checkpoint_ensemble_weights.py) |
| Joint fusion | 5.6.2, 6.5.2–6.5.4 | Six trainer/model/extractor families below |
| Evaluation | 5.7 | [Validation-thresholded F1](../scripts/evaluate_zero_shot_f1_from_validation.py), [pointing-game notebook](../notebooks/localization_pointing_game.ipynb) |

## Guidance run identity

A1, A2, A3, A3.1, A3.2 and A4 each require separate Cardiomegaly and Pleural Effusion run records. A0 is the original-image reference, not automatically the same checkpoint across every historical trainer.

| Variant | Configuration evidence | Caveat |
|---|---|---|
| A0 | [A0 YAML](../configs/Pretrain_anatomy_prior_A0.yaml) | Earlier trainer ignored subset key; reported run used full dataset |
| A1 | [A1 YAML](../configs/Pretrain_anatomy_prior_A1.yaml) | Same full-data clarification; named file primarily describes Cardiomegaly |
| A2 | [A2 YAML](../configs/Pretrain_anatomy_prior_A2.yaml) | Same full-data clarification; retain uncertainty semantics |
| A3 | [Current A3 YAML](../configs/Pretrain_anatomy_prior_A3.yaml) | Current file represents later segmentation-derived prior, not thesis rectangular support |
| A3.1 | Historical commit `ea61e1e` | Coefficient 0.005; changes same A3 filename; full-data selection |
| A3.2 | Historical commit `c9a5803` | Coefficient 0.001; changes same A3 filename |
| A4 | [A4 YAML](../configs/Pretrain_anatomy_prior_A4.yaml) | Preservation coefficient has historical 0.5, 0.1 and 1.0 variants |

The thesis's final run configs, checkpoint-selection records and PE-specific commands still need recovery. No existing file is relabeled as a verified reproduction config in this phase.

## Fusion families

| Family | Trainer | Config |
|---|---|---|
| FC | [Trainer](../Pretrain_A0_multiview_fusion_dual_auc.py) | [Config](../configs_new/Pretrain_A0_multiview_fusion_dual_auc.yaml) |
| Mean | [Trainer](../Pretrain_A0_multiview_mean_fusion_dual_auc.py) | [Config](../configs_new/Pretrain_A0_multiview_mean_fusion_dual_auc.yaml) |
| Gated | [Trainer](../Pretrain_A0_multiview_gated_fusion_dual_auc.py) | [Config](../configs_new/Pretrain_A0_multiview_gated_fusion_dual_auc.yaml) |
| Residual | [Trainer](../Pretrain_A0_multiview_residual_fusion_dual_auc.py) | [Config](../configs_new/Pretrain_A0_multiview_residual_fusion_dual_auc.yaml) |
| Transformer | [Trainer](../Pretrain_A0_multiview_transformer_fusion_dual_auc.py) | [Config](../configs_new/Pretrain_A0_multiview_transformer2_mean_dual_auc.yaml) |
| Cross-attention | [Trainer](../Pretrain_A0_multiview_cross_attention_fusion_dual_auc.py) | [Config](../configs_new/Pretrain_A0_multiview_cross_attention_fusion_dual_auc.yaml) |

Each family has a separately named `scripts_new/extract_<family>_fusion_itc_margin_gradcam_heatmaps.py`. Their model definitions are linked in [methodology](methodology.md). Keep the three initial single-view checkpoint identities and selection families explicit.

The file [zero_shot_eval_vindr_multiview_pos_neg_bare_prompts.py](../scripts_new/zero_shot_eval_vindr_multiview_pos_neg_bare_prompts.py) currently imports the cross-attention model directly despite its generic name. It is not a verified dispatcher for all six architectures.

## External references and historical work

- BioViL-T: [native extraction](../scripts_new/extract_biovil_t_phrase_grounding.py) and [same-ALBEF-case extraction](../scripts_new/extract_biovil_t_same_albef_cases.py).
- CheXzero: [comparison notebook](../notebooks/visualize_chexzero_checkpoints_and_model_comparison.ipynb) consumes externally generated maps; complete extraction/classification code is absent.
- A5/A6: preserve their trainers, configs, patch-head module, extractors and dependent notebooks as non-thesis history. No deletion or archival move has occurred.
- FROC, CLARity-style gating and older attribution variants: preserve as exploratory localization. They are not the primary thesis pointing-game endpoint.
- VQA, NLVR, retrieval, visual entailment and RefCOCO: inherited ALBEF functionality; preserve source and attribution.

A notebook covering A1–A5 remains historical as a whole; do not remove A5 cells simply to match main-thesis navigation.
