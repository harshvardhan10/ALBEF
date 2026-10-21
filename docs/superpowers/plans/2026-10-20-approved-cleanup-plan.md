> Approved by the user on 2026-10-20. Consolidation, ITC-margin method and historical narrative/environment confirmed. Cross-attention training completed; checkpoint location confirmed. Exact training commit and resolved config remain unconfirmed. Original proposed plan follows unchanged.

# Thesis Repository Cleanup Plan

**Date:** 2026-10-20
**Status:** Proposed; awaiting your review and explicit approval. No repository cleanup has started.
**Goal:** Give the thesis supervisor one repository in which the experimental narrative, original contributions, model differences, launch commands, and validation evidence are easy to review.
**Execution:** Work sequentially in this chat after approval. Use `superpowers:executing-plans` task by task; do not dispatch subagents by default. Preserve progress in a compact checklist so work can resume without repeating the audit.
**Architecture:** Use ALBEF as the consolidated repository. Import selected cluster workflows and a pinned CheXzero baseline into clearly separated directories. Initially retain the root Python packages and historical entry points to protect cluster imports and checkpoints.
**Tech stack:** Existing Python/PyTorch research code, YAML and SLURM. The earlier design records Python 3.8.20, PyTorch 1.8.1+cu111 and CUDA 11.1; these are historical requirements to confirm, not a newly verified environment.

## 1. Evidence and what has already been done

Read-only GitHub inspection found:

| Repository | Inspected revision | Role |
|---|---|---|
| [ALBEF](https://github.com/harshvardhan10/ALBEF/tree/thesis-repository-cleanup) | `6320a2e93ce63bf2a77f3c716278c4379e9114f1` | Main code; cleanup branch |
| [thesis-codebase](https://github.com/harshvardhan10/thesis-codebase) | `4b189fa90d8aa18be1ce22d003e908988cfa79b6` | SLURM jobs, preparation scripts and notebooks |
| [CheXzero](https://github.com/harshvardhan10/CheXzero) | `5f3997d8456fc2fe75742018557d79793cc546aa` | Zero-shot baseline and attention Grad-CAM |

At inspection, ALBEF's cleanup branch is six commits ahead of `main`, zero behind. The entire diff consists of four added planning documents. The model/trainer consolidation and repository hygiene described there have not landed on this branch. This does not establish what ran on the server.

Existing requirements reviewed:

- [Design](https://github.com/harshvardhan10/ALBEF/blob/thesis-repository-cleanup/docs/superpowers/specs/2026-09-17-thesis-repository-cleanup-design.md)
- [Navigation and hygiene plan](https://github.com/harshvardhan10/ALBEF/blob/thesis-repository-cleanup/docs/superpowers/plans/2026-09-17-repository-navigation-hygiene.md)
- [Model consolidation plan](https://github.com/harshvardhan10/ALBEF/blob/thesis-repository-cleanup/docs/superpowers/plans/2026-09-17-multiview-model-consolidation.md)
- [Training and evaluation plan](https://github.com/harshvardhan10/ALBEF/blob/thesis-repository-cleanup/docs/superpowers/plans/2026-09-17-training-evaluation-consolidation.md)

Those documents are useful requirements, not verified implementation instructions. In particular, their example tests use `torch.testing.assert_close`, whose availability cannot be assumed in the recorded PyTorch 1.8.1 environment. Use supported numerical assertions and real behavioral checks rather than source-text/count tests as proof of equivalence.

## 2. Decisions to confirm before execution

Approval of this plan should explicitly confirm or correct these points:

1. **Consolidation:** approve ALBEF as the single review repository, with imported cluster scripts and a pinned CheXzero snapshot. This replaces the September design's explicit decision to keep both external.
2. **Scientific narrative:** retain the six stages below, the distinction between qualitative localization and quantitative classification, and exploratory FROC status. Has any of that changed since September?
3. **Cross-attention status:** has training finished? Provide the actual training commit, resolved run config, available checkpoint roles and checkpoint paths. Confirm whether both `checkpoint_best_cardiomegaly_auc.pth` and `checkpoint_best_macro_auc_stable.pth` exist.
4. **Heatmap method:** proposed validation uses the fused ITC positive-minus-negative bare-prompt score and ViT self-attention Grad-CAM for each branch. Confirm this is the intended comparison method, rather than text-image cross-attention Grad-CAM or direct attention inside the fusion module.
5. **Environment and data:** confirm the working Python/PyTorch/CUDA environment, tokenizer cache, VinDr image/label paths and lung/heart mask roots. Supply a small agreed case list and any existing reference heatmaps; otherwise select fixed cases before inspecting new outputs.
6. **Related work:** CheXzero is the supplied baseline repository. If a separate literature/related-work repository was intended, provide its URL before claiming consolidation is complete.

Unknown experimental outcomes will remain explicitly unconfirmed. No results will be inferred from filenames or old prose.

## 3. Boundaries and recommended consolidation

- All eventual changes belong on `thesis-repository-cleanup`. Leave `main` and the source repositories untouched. No history rewriting, force pushes, merges, remote deletions, new account connections, messages or publishing.
- This stage creates only this local plan. Approval is required before cloning for implementation or changing repository content. A push or PR is a separate delivery decision, not implicit in plan approval.
- Preserve existing files during planning. During approved cleanup, propose tracked-file removals in a concrete inventory; obtain specific approval before destructive removals not already approved. Untracking a data manifest must preserve the user's local data.
- Recommend a **curated source import**, with source URL, exact revision, original path, destination, modifications and license recorded in `docs/source_provenance.md`. Avoid submodules because the requested outcome is one checkout; avoid merging unrelated Git histories because it adds review noise. Source histories remain accessible at their original repositories.
- Import required CheXzero source under `baselines/chexzero/`, preserving `LICENSE` and `NOTICE.md`. Keep its environment separate; check local imports such as `model`, `clip` and `eval` resolve within that baseline. Retain any required tokenizer vocabulary.
- Import SLURM workflows under `cluster/slurm/` and supporting preparation scripts under `cluster/data_preparation/`. Preserve relevant preparation notebooks under `notebooks/data_preparation/` with provenance. Inventory every source file so excluded logs/download helpers are intentional rather than silently lost.
- Do not import generated logs, raw medical data, weights, predictions or notebook outputs into the review repository. Record their roles and external storage requirements instead.
- Preserve ALBEF attribution. Do not relicense imported code. The thesis-codebase metadata reports no declared license; record authorship/provenance and resolve any third-party notices before redistribution.
- Defer the September design's broad `src/cxr_albef/` migration. The existing root packages are the safest first consolidation boundary.

Proposed additions (existing root code remains initially):

```text
ALBEF/
  docs/                  # narrative, provenance, reproduction and validation
  models/fusion/         # six distinct fusion implementations
  training/              # shared helpers and multiview trainer
  evaluation/            # checkpoint roles and result metadata
  localization/          # method metadata and shared extraction utilities
  cluster/slurm/         # imported, portable launch workflows
  cluster/data_preparation/
  baselines/chexzero/    # pinned baseline with original notices
  configs/experiments/   # methodological configs
  configs/environments/ # examples; real machine paths stay local
  tests/                 # focused behavior and compatibility checks
  results/summaries/     # curated evidence only
  legacy/                # scientifically relevant superseded code
```

## 4. Work sequence after approval

### Task 1 — Record a resumable baseline and review map

- [ ] Obtain a clean local checkout of the existing cleanup branch; record the current branch heads and source revisions before changing anything. Reconcile any new upstream changes with this plan.
- [ ] Inventory scripts, configs, imports, source licenses, datasets, outputs and duplicates. Classify each file as canonical, compatibility entry point, historical, generated or excluded-from-import. Do not infer unused status merely from absence of imports: SLURM and notebooks can invoke scripts.
- [ ] Create `docs/cleanup_inventory.md`, `docs/source_provenance.md` and a compact progress record under `docs/superpowers/`. Record old-to-new paths and planned removals.
- [ ] Verify the checkout is on `thesis-repository-cleanup`, source repositories are unchanged, and every supplied repository has an inclusion/exclusion record.

### Task 2 — Make the thesis reviewable before refactoring

**Files:** `README.md`; `docs/experiment_catalogue.md`, `checkpoints.md`, `reproduction.md`, `limitations.md`, `model_architecture.md`, `localization_methods.md` and `data_preparation.md`.

- [ ] Organize the catalogue around: (1) ALBEF/BioViL-T/CheXzero baselines, (2) A0–A4 anatomy guidance, (3) A5/A6.1 patch refinement, (4) original/lung/heart single-view models, (5) score/heatmap/learned ensembles and six joint fusion variants, (6) evaluation and failure analysis.
- [ ] Map each experiment to hypothesis, original contribution versus upstream code, model, trainer, config, launch job, checkpoint role, evaluation command, available evidence and status.
- [ ] Preserve the September scientific constraints unless corrected: no invented semi-supervised contribution, no final quantitative localization ranking, and no completed status for unconfirmed cross-attention results. Historical findings must be attributed to the old design until supported by supplied results.
- [ ] Verify internal links and paths; make it possible to find the cross-attention architecture and reproduction route directly from the README.

### Task 3 — Consolidate cluster workflows and the CheXzero baseline

- [ ] Import the curated files at recorded source revisions. Retain a source-to-destination map for all exclusions and renamed scripts.
- [ ] Replace fixed checkout/data/output paths with documented environment inputs. Derive ALBEF's root from the script location. Preserve job resources and scientific arguments. Remove default email notification directives from portable examples; no jobs are submitted by the agent.
- [ ] Keep CheXzero dependencies isolated from ALBEF; document its launch directory and ensure its tokenizer and local imports do not collide with ALBEF modules.
- [ ] Reconcile `scripts/` versus `scripts_new/` by responsibility with compatibility wrappers where needed. Check every imported launcher points to a real script/config in the consolidated checkout.
- [ ] Review shell syntax with `bash -n` where available and inspect dependencies between jobs; do not run download, training or submission scripts.

### Task 4 — Establish cross-attention heatmap validation before changing models

**New files:** `scripts_new/extract_cross_attention_fusion_itc_margin_gradcam_heatmaps.py`, `cluster/slurm/extract_cross_attention_fusion_heatmaps.slurm`, `docs/cross_attention_validation.md`, focused extraction tests.

- [ ] Freeze the exact training model/config provenance. Build a baseline extractor against the existing cross-attention fusion model before consolidation.
- [ ] Adapt the already inspected Transformer ITC-margin extraction workflow, explicitly importing `models.model_pretrain_multiview_cross_attention_fusion.ALBEF`. Retain its controlled three-view preprocessing and score definition, but correct all model/method metadata, output names and fusion descriptions. This is a new implementation, not an existing working command.
- [ ] Require strict checkpoint loading; fail on missing/unexpected keys. Do not silently load a single-view checkpoint or discard fusion parameters. Capture gradients through the actual anchored fusion computation.
- [ ] Preserve CLI flags shown in section 5, output branch CAMs and composite CAMs, and write a manifest containing revision, checkpoint hash/role, resolved config, case IDs, prompts, layer, normalization and environment. Reject incompatible inputs; never overwrite old validation outputs by default.
- [ ] Add tests for wrong checkpoint architecture, missing/misaligned masks, image selection, finite scores/gradients, normalization of zero maps, hook cleanup and method metadata. Static inspection cannot establish these runtime properties.
- [ ] Deliver the baseline server instructions in section 5. The user generates baseline evidence before model consolidation; if that cannot happen yet, model compatibility remains unverified and the later refactor gate stays open.

### Task 5 — Consolidate the six multiview models without changing the experiment

**Files:** `models/fusion/{factory,fc,mean,gated,residual,transformer,cross_attention}.py`, `models/model_pretrain_multiview.py`, six legacy model modules, `tests/models/`.

- [ ] Extract each actual fusion implementation without redesigning it. Preserve FC initialization, mean averaging, sample-specific gates, original-anchored residuals, Transformer token processing and anchored cross-attention ordering.
- [ ] Compare all six full models before selecting a shared body. Do not assume that non-fusion code is identical. Preserve online/momentum encoders, queue, objectives, negative sampling, public `ALBEF` constructor and existing state-dict names.
- [ ] Preserve `view_fusion.*` and `view_fusion_m.*` and strict loading of existing checkpoints. Avoid a key mapper unless unavoidable, documented and tested against real historical schemas.
- [ ] Keep historical import paths as wrappers. Freeze independent legacy reference implementations at recorded revisions so equivalence tests do not compare a wrapper to itself.
- [ ] Check deterministic output and gradient equivalence, state keys/shapes, momentum updates and wrapper behavior for all variants. Any unavailable runtime checks remain marked not run.
- [ ] Repeat the user's fixed-case cross-attention heatmap run after refactoring and compare against the baseline before accepting this task.

### Task 6 — Consolidate training, data alignment and configs

**Files:** `dataset/multiview_cxr_dataset.py`; `training/{multiview_checkpoints,multiview_trainer,anatomy_guidance,patch_refinement}.py`; `Pretrain_A0_multiview.py`; historical trainers; `configs/experiments/`; environment examples.

- [ ] Share synchronized three-view dataset handling and checkpoint initialization. Test identity/caption alignment, joint transforms, missing masks, encoder initialization and resumption behavior.
- [ ] Consolidate multiview training with explicit fusion selection, preserving existing CLI paths, checkpoint selection, validation frequency, loss behavior and distributed semantics.
- [ ] Extract genuinely shared A0–A4 helpers and A5/A6.1 helpers without collapsing their distinct objectives or trainable/frozen behavior.
- [ ] Create immutable methodological configs and separate machine-path inputs. Save resolved config with each future run. Preserve historical configs as evidence; do not retroactively edit them to claim full-dataset runs. Use `train_subset_size: null` for intended full runs and separate bounded smoke configs.
- [ ] Verify checkpoint names, momentum state, gradient routing, resolved-config precedence and legacy launcher arguments. Heatmap validation alone cannot certify training/DDP equivalence: document optional user-run one-batch checks for each modified workflow.

### Task 7 — Organize evaluation and perform approved hygiene

**Files:** `evaluation/{experiment_metadata,checkpoint_roles}.py`, `localization/methods.py`, affected runners, `.gitignore`, dependency files, notebook index, legacy inventory and example manifests.

- [ ] Record label order, prompt definitions, split, checkpoint role, threshold/ensemble fitting source and method details next to outputs. Fit thresholds/ensemble weights on validation only; preserve held-out test boundaries.
- [ ] Keep FROC exploratory and preserve its command entry points if moved. Treat heatmap methods as distinct even when images look similar.
- [ ] Remove only approved generated/duplicate tracked artifacts after references are checked. Preserve local data; use synthetic example manifests and preparation instructions. Strip notebook outputs selectively, preserving curated scientific evidence elsewhere.
- [ ] Separate runtime/evaluation/notebook/development dependencies without upgrading the successful training stack by default. Inspect syntax compatibility: existing extractors use modern annotations, so imports in the actual Python 3.8 environment need explicit checking.
- [ ] Verify notebook readability, dependency declarations, example schemas, wrapper resolution and absence of accidental data/weights/log additions. Do not rewrite repository history.

### Task 8 — Deliver a supervisor review package

- [ ] Review the full diff against the recorded starting cleanup commit; separate organizational changes from behavioral changes and unexplained differences.
- [ ] Record local inspection/check results honestly in `docs/cleanup_validation.md`; list every unavailable check and user-run result separately. Do not install a large GPU stack just to obtain a green local test report.
- [ ] Produce `docs/supervisor_review_guide.md` with a recommended reading order, architecture differences, important changed paths and evidence limitations.
- [ ] Leave changes on `thesis-repository-cleanup`. Do not merge into or update `main`. Discuss push/PR delivery separately.

## 5. User-run cross-attention validation contract

### What exists today and what must be added

`scripts/extract_crossattn_gradcam_heatmaps.py` uses the single-view model builder in `scripts/src.py`; its name refers to text-image cross-attention. It must not be used as proof that multiview cross-attention fusion loads correctly.

`scripts_new/extract_transformer_fusion_itc_margin_gradcam_heatmaps.py` explicitly loads the Transformer fusion model, not the cross-attention fusion model. Its CLI and methodology provide a starting point, but swapping only its YAML/checkpoint is not valid.

The following commands are the **required interface of the new extractor from Task 4**, and are not runnable against the currently inspected branch. Final instructions will substitute the user's confirmed paths and actual output schema after implementation.

### Inputs and preparation

1. Use a separate server checkout of `thesis-repository-cleanup`; do not alter a directory used by a running job. Record `git rev-parse HEAD` and the training checkpoint's source commit.
2. Activate the known working environment on an allocated GPU node. Record Python, torch, CUDA, NumPy, Pillow and torchvision versions. Confirm the tokenizer is cached locally; do not rely on downloading it during a job.
3. Prepare a fixed small CSV with the canonical image-ID column first and label columns including `Cardiomegaly` and `Pleural effusion`. Select approximately 4–8 cases with positives for both pathologies and negative controls before examining new maps. The identical CSV must be used before and after refactoring.
4. Verify every selected `<images_root>/<image_id>.png` and both mask paths `<mask_root>/<image_id[:2]>/<image_id>.png` exist and agree in dimensions. The imported workflow must match mask compositing/resizing to the trained model's preprocessing, not introduce a new convention.
5. Use the resolved cross-attention training config and an actual cross-attention fusion checkpoint. The current training launcher suggests outputs under `outputs/A0_multiview_cross_attention_fusion_dual_auc/`; file existence and completion are unconfirmed.

### Extraction command after Task 4

Run from the ALBEF checkout root, on the allocated GPU node. Replace all angle-bracket values with confirmed local paths. Use distinct output directories for baseline/refactored runs and for each checkpoint role.

```bash
python scripts_new/extract_cross_attention_fusion_itc_margin_gradcam_heatmaps.py \
  --config '<resolved-cross-attention-config.yaml>' \
  --checkpoint '<checkpoint_best_cardiomegaly_auc.pth>' \
  --labels_csv '<fixed-validation-cases.csv>' \
  --images_root '<vindr-test-images>' \
  --lung_mask_root '<vindr-test-lung-mask-root>' \
  --heart_mask_root '<vindr-test-heart-mask-root>' \
  --tokenizer_name_or_path '<local-bert-tokenizer-snapshot>' \
  --output_dir '<new-empty-baseline-or-refactored-output-directory>' \
  --target_labels Cardiomegaly 'Pleural effusion' \
  --vit_layer -2 \
  --device cuda
```

Repeat for `checkpoint_best_macro_auc_stable.pth` if available, using a separate directory. Never pass `--overwrite` for a comparison run. If submitted through SLURM, the delivered job must expose these same inputs, create the log directory before submission and cap the smoke run to the fixed CSV. The agent does not submit it.

### Acceptance checks and evidence to return

- [ ] Strict checkpoint load succeeds with no missing/unexpected keys, and metadata explicitly identifies anchored cross-attention fusion with original anchor and ordered lung/heart memories.
- [ ] There is exactly one successful `.pt` result per selected case, an index CSV and a manifest; no case was silently skipped or substituted.
- [ ] Both requested labels have finite scores and original/lung/heart/composite maps. Captured gradients exist. All-zero maps are reported for investigation rather than described as successful localization.
- [ ] Image resolution and patch-grid dimensions match the model/config. Vis maps lie in `[0,1]`; raw signed evidence need not. Metadata records the resolved zero-based layer (`-2` means second-to-last block).
- [ ] Use the same checkpoint bytes, cases, config, prompts, preprocessing and environment before/after refactoring. Compare raw score and raw branch/composite CAM tensors, not only normalized screenshots. Deliver a comparison utility with per-field max absolute differences; start with `rtol=1e-5, atol=1e-6` for controlled FP32 runs, and investigate failures rather than silently loosening tolerance. These are proposed thresholds, not observed equivalence.
- [ ] Save matched overlays using identical display scaling and case ordering. Technical equivalence does not require visually good localization; report poor localization honestly.
- [ ] Return the command, branch commit, checkpoint SHA-256/role, environment versions, full job log, manifest, comparison report and a few agreed overlays. Keep medical images and credentials on the server; share only permitted artifacts.

If no baseline can be generated, label the result a post-cleanup smoke test only. It cannot establish old-versus-new equivalence. If a run fails, stop broadening the cleanup and investigate that specific failure.

## 6. Credit and progress controls

- Work sequentially and reuse this audit. No default subagent reviews, repeated full-repository scans, model upgrades or local full experiments.
- Prioritize the supervisor-facing map, consolidation and cross-attention validation path. Structural refactoring proceeds only with compatibility evidence; do not replace it with superficial source-text tests to save effort.
- Save a progress checkpoint after each accepted task: changed files, evidence, failures, remaining work and the next action. Keep commits small and scoped when committing is authorized.
- Exact credit consumption cannot be guaranteed or capped by this plan. If budget becomes tight, preserve completed work and an honest continuation record; never mark incomplete or unvalidated work complete. The user has authorized using the available budget to finish, not skipping validation.
- Server execution is the user's responsibility. Implementation can be ready for validation while final runtime acceptance remains pending the user's evidence.

## 7. Final completion checklist

- [ ] User approved this plan and resolved the decisions in section 2.
- [ ] All changes are isolated to the cleanup branch; `main` and source repositories are untouched.
- [ ] One checkout contains the agreed thesis code, baseline code and orchestration workflows, with source/license provenance and explicit exclusions.
- [ ] The supervisor can follow every thesis stage from README to implementation, config, launch command, checkpoint and supported evidence.
- [ ] Fusion variants remain distinct and checkpoint/import/CLI compatibility is verified or explicitly pending.
- [ ] Approved hygiene is complete without losing local data or scientifically relevant history.
- [ ] Precise, implemented cross-attention heatmap instructions and comparison tooling have been delivered.
- [ ] The user has run the baseline/refactored checks and reviewed the evidence; remaining training/DDP limitations are explicit.
- [ ] No unsupported scientific conclusions or unperformed test claims appear in the documentation.

**Approval gate:** Review this file and answer section 2. No cleanup implementation begins until you explicitly approve the resulting scope.
