# Experiment catalogue

Statuses reflect the September design reconfirmed by the author. The cleanup has not independently rerun any experiment. “Evaluation pending” means this checkout does not yet contain verified evidence for the relevant checkpoint roles. The two roles are best_cardiomegaly_auc and best_macro_auc_stable; last checkpoints are recovery-only.

## Stage 1 — Recognition and localization baselines

- ALBEF: models/model_pretrain.py, scripts/zero_shot_eval_vindr.py and scripts_new/extract_albef_itc_gradcam_heatmaps.py.
- BioViL-T: scripts_new/extract_biovil_t_phrase_grounding.py and extract_biovil_t_same_albef_cases.py; original launch records in cluster/historical/scripts_new.
- CheXzero: baselines/chexzero/zero_shot_vindr.py and extract_chexzero_multi_checkpoint_attention_gradcam.py at the pinned source revision. Launch records include cluster/historical/scripts/chexzero_eval_zero_shot_vindr_a40.slurm and scripts_new/extract_chexzero_attention_gradcam.slurm.
- Hypothesis: establish recognition and pathology-localization reference behavior. Curated comparable result evidence remains pending; do not describe the baseline imports as new contributions.

## Stage 2 — Direct anatomical guidance (A0–A4)

- Pretrain_anatomy_prior.py: A0 control, A1 support, A2 uncertainty-weighted support, A3 positive-only support. Matching configs: configs/Pretrain_anatomy_prior_A0.yaml through _A3.yaml. Launch records: cluster/historical/scripts/run_A0_anatomy_baseline.slurm, run_A1_anatomy_support.slurm through run_A3_anatomy_support.slurm.
- Pretrain_anatomy_prior_A4.py: positive-only guidance with classification preservation, configs/Pretrain_anatomy_prior_A4.yaml, cluster/historical/scripts/run_A4.slurm.
- Hypothesis: guide spatial evidence without losing recognition. Historical runs described in the design; checkpoint-role evaluation coverage remains pending curation.

## Stage 3 — Learned spatial refinement

- A5: Pretrain_anatomy_prior_A5_patch_head.py and configs/Pretrain_A5_patch_head.yaml. Train ALBEF with standard objectives while a patch head uses detached attention.
- A6.1: Pretrain_anatomy_prior_A6_1_frozen_A0_patch_head.py and matching configs/Pretrain_A6_1_frozen_A0_patch_head.yaml. ALBEF frozen, patch head trainable.
- Launch records: cluster/historical/scripts/pretrain_A5_patch_head.slurm and pretrain_A6_1_frozen_A0_patch_head.slurm. Extraction: scripts/extract_patch_head_heatmaps.py.
- Hypothesis: pathology-specific spatial refinement rather than merely reproducing generic anatomical support. Failure analysis belongs here; no semi-supervised claim is introduced.

## Stage 4 — Anatomy-focused single views

Pretrain_A0_maskcache.py with configs/Pretrain_A0_original_dual_auc.yaml, Pretrain_A0_lung_dual_auc.yaml and Pretrain_A0_heart_dual_auc.yaml. Inspect the actual launch record and saved config because multiple historical preprocessing variants coexist. cluster/historical/scripts/run_vindr_three_views*.slurm records evaluation routes. Original/lung/heart inputs require matched cases, masks and prompts.

## Stage 5 — View combinations

Prediction ensembles: scripts/ensemble_zero_shot_scores.py. Heatmap ensembles: scripts/ensemble_crossattn_gradcam_heatmaps.py. Validation-fitted ensembles: scripts_new/learn_dual_checkpoint_ensemble_weights.py. Fit thresholds/weights on validation only; test metrics must not refit them.

Joint fusion variants use separate model modules listed in [model_architecture.md](model_architecture.md):

| Fusion | Trainer | Config | Historical orchestration | Status |
|---|---|---|---|---|
| fc | [Pretrain_A0_multiview_fusion_dual_auc.py](../Pretrain_A0_multiview_fusion_dual_auc.py) | [configs_new/Pretrain_A0_multiview_fusion_dual_auc.yaml](../configs_new/Pretrain_A0_multiview_fusion_dual_auc.yaml) | [launch record](../cluster/historical/scripts_new/pretrain_multiview_fusion.slurm) | historical experiment; evaluation evidence pending curation |
| mean | [Pretrain_A0_multiview_mean_fusion_dual_auc.py](../Pretrain_A0_multiview_mean_fusion_dual_auc.py) | [configs_new/Pretrain_A0_multiview_mean_fusion_dual_auc.yaml](../configs_new/Pretrain_A0_multiview_mean_fusion_dual_auc.yaml) | [launch record](../cluster/historical/scripts_new/pretrain_multiview_mean_fusion.slurm) | historical experiment; evaluation evidence pending curation |
| gated | [Pretrain_A0_multiview_gated_fusion_dual_auc.py](../Pretrain_A0_multiview_gated_fusion_dual_auc.py) | [configs_new/Pretrain_A0_multiview_gated_fusion_dual_auc.yaml](../configs_new/Pretrain_A0_multiview_gated_fusion_dual_auc.yaml) | [launch record](../cluster/historical/scripts_new/pretrain_multiview_gated_fusion.slurm) | historical experiment; evaluation evidence pending curation |
| residual | [Pretrain_A0_multiview_residual_fusion_dual_auc.py](../Pretrain_A0_multiview_residual_fusion_dual_auc.py) | [configs_new/Pretrain_A0_multiview_residual_fusion_dual_auc.yaml](../configs_new/Pretrain_A0_multiview_residual_fusion_dual_auc.yaml) | [launch record](../cluster/historical/scripts_new/pretrain_multiview_residual_fusion.slurm) | historical experiment; evaluation evidence pending curation |
| transformer | [Pretrain_A0_multiview_transformer_fusion_dual_auc.py](../Pretrain_A0_multiview_transformer_fusion_dual_auc.py) | [configs_new/Pretrain_A0_multiview_transformer2_mean_dual_auc.yaml](../configs_new/Pretrain_A0_multiview_transformer2_mean_dual_auc.yaml) | [launch record](../cluster/historical/scripts_new/pretrain_multiview_transformer2_mean.slurm) | historical experiment; evaluation evidence pending curation |
| cross_attention | [Pretrain_A0_multiview_cross_attention_fusion_dual_auc.py](../Pretrain_A0_multiview_cross_attention_fusion_dual_auc.py) | [configs_new/Pretrain_A0_multiview_cross_attention_fusion_dual_auc.yaml](../configs_new/Pretrain_A0_multiview_cross_attention_fusion_dual_auc.yaml) | [launch record](../cluster/historical/scripts_new/pretrain_multiview_cross_attention_fusion.slurm) | training complete; evaluation pending (author confirmed) |

For all six: compare cardiomegaly and stable-label macro checkpoints separately. The cross-attention files are reported on the server; exact training commit is unconfirmed; the saved run YAML was compared by the author and differs only in comments. The new [validation guide](cross_attention_validation.md) produces fixed-case qualitative maps and technical baseline evidence, not a localization ranking.

## Stage 6 — Evaluation and interpretation

Classification: scripts/vindr_classification_validation.py, scripts_new/vindr_multiview_classification_validation.py and scripts/evaluate_zero_shot_f1_from_validation.py. Record label order, prompts, split, checkpoint role and validation-selected thresholds. Keep qualitative localization comparisons matched by case and method. FROC implementations now live in scripts/exploratory_froc/, with compatibility commands at their original scripts/ paths. They remain exploratory. See [limitations](limitations.md).

## Canonical cleanup entry points

All six multiview trainer paths above select `training/multiview_trainer.py`; new runs may use `Pretrain_A0_multiview.py --fusion_type NAME`. `dataset/multiview_cxr_dataset.py` owns synchronized three-view transforms/alignment. `training/multiview_checkpoints.py` owns single-view initialization and full resume. Their original names remain available through the old entry points.

Portable intended-full-data presets are under `configs/experiments/stage_02_anatomy_guidance/`, `stage_03_patch_refinement/`, `stage_04_single_view/` and `stage_05_multiview/`. They retain methodological settings but require private path overlays. Historical YAML is unchanged and remains the source of run provenance. `configs/smoke/` is diagnostic only. See [cluster workflows](cluster_workflows.md) for commands and limits.

`evaluation/checkpoint_roles.py` retains independent cardiomegaly/macro selection and recovery-only last checkpoints. `evaluation/artifacts.py` writes unchanged score arrays with separate metadata sidecars for future shared-trainer outputs. `evaluation/albef_utils.py` consolidates the formerly identical scripts/src.py and scripts_new/src.py; both import paths remain usable. Validation-fitted ensembles record input hashes and retain validation/test boundaries.
