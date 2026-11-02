# Reproduction guide

Run commands from the repository root. The checkout now has a canonical research hierarchy plus legacy compatibility paths. It is not yet a complete portable reproduction package. Missing final configs, assets and selection records are listed in [discrepancies](discrepancies.md). No GPU training or full thesis result has been reproduced by the documentation or structural migration phases.

## Canonical entry points and compatibility

Run from the repository root, after activating the appropriate existing environment:

```bash
python -B tools/run_experiment.py experiments/baseline/Pretrain.py --help
python -B tools/run_experiment.py experiments/anatomy_guidance/Pretrain_anatomy_prior.py --help
python -B tools/run_experiment.py experiments/localization/itc/extract_itc_margin_gradcam_heatmaps.py --help
python -B -m unittest discover -s tests/compatibility -v
```

The launcher accepts canonical or legacy paths listed as entry points in [the applied manifest](audit/applied_migration.csv), restores historical imports/arguments/`__file__` context and forwards remaining flags unchanged. It rejects unknown paths and invocation outside the repository working directory. Do not substitute direct deep-path execution of canonical files. Legacy commands below remain supported through compatibility loaders; config values/default paths remain unchanged via relative symlinks.

Reusable in-checkout imports need both root and source paths, for example `PYTHONPATH="$PWD:$PWD/src"`. There is no new standalone package installation in this phase. Symlink support is required for old config paths (validated on Linux). Seven output-free notebooks have canonical viewing locations and legacy symlinks; the others remain in place. All notebook cells/outputs are unchanged. Inspect [notebook context](artifact_organization.md) before interactive execution.

## Choose the workflow environment

Separate legacy ALBEF, newer research/fusion, BioViL-T, anatomical segmentation and analysis environments where needed. The [dependency inventory](external_dependencies.md) records existing evidence. Do not replace historical versions with guessed current releases.

The pointing-game notebook declares Python 3.10+; it is not covered by the Python 3.9 legacy smoke recipe below. Newer validation transforms require `Image.Resampling`, absent from legacy Pillow 8.4. The local XBERT relies on historical Transformers APIs, so solving that incompatibility by upgrading the whole stack is not a verified correction.

## Legacy CPU smoke installation

The following versions were validated for the original CPU model components during onboarding: Python 3.9.25, torch 1.8.0+cpu, torchvision 0.9.0+cpu and the existing legacy requirements. This does not recover the thesis's two-A40 training environment.

With `uv` available, choose a new writable directory outside the checkout:

```bash
# Set this to a new writable directory; do not reuse an unrelated environment.
SETUP_ROOT=/absolute/path/to/new-albef-setup
mkdir -p "$SETUP_ROOT"
export UV_PYTHON_INSTALL_DIR="$SETUP_ROOT/python"
export UV_PYTHON_BIN_DIR="$SETUP_ROOT/bin"
export UV_CACHE_DIR="$SETUP_ROOT/cache"
uv python install 3.9.25
uv venv --python 3.9.25 "$SETUP_ROOT/venv"
uv pip install --python "$SETUP_ROOT/venv/bin/python" \
  torch==1.8.0+cpu torchvision==0.9.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$SETUP_ROOT/venv/bin/python" \
  -r requirements_albef_legacy.txt
uv pip check --python "$SETUP_ROOT/venv/bin/python"
source "$SETUP_ROOT/venv/bin/activate"
python -B Pretrain.py --help
```

Execute from the checkout root; network access to Python release/package hosts and PyTorch wheels is required. Keep TLS verification enabled. Requirements include historical unpinned transitive dependencies, so this recipe is not a frozen training lock. Current-instance CLI/help checks are recorded in the [validation record](validation.md); no fresh-machine restoration claim is made.

Onboarding also exercised synthetic image augmentation, a small vision encoder, XBERT fusion, masked-language loss and backpropagation. Its helper lives outside the checkout and is not a portable repository test suite.

## Prepare data and assets

Follow [data access](data_access.md). Recover the exact training/common-valid manifest, frozen validation IDs/labels, official test manifest/annotations/dimensions, and required mask caches. Verify non-overlap and membership from artifacts rather than directory names.

Recover initial and selected checkpoint hashes and the actual BERT tokenizer/model snapshot. `scripts/src.py` and its duplicate hardcode a private local-only tokenizer cache; commands consuming that loader are blocked on this portability issue until separately addressed. Do not create a fake cache or replace tokenizer behavior silently.

Keep private configs and datasets outside the checkout. Portable data/checkpoint/tokenizer/output-root overrides are planned but not implemented in this phase.

## Training entry points

Start with [experiment mappings](experiments.md). The following are argument-inspection commands, not training validation:

```bash
python -B Pretrain.py --help
python -B Pretrain_anatomy_prior.py --help
python -B Pretrain_anatomy_prior_A4.py --help
python -B Pretrain_A0_multiview_dual_auc.py --help
python -B Pretrain_A0_multiview_cross_attention_fusion_dual_auc.py --help
```

Actual thesis training needs the exact run YAML, input assets, matching GPU stack and launch command. The thesis reports two A40 GPUs, maximum 30 epochs, AdamW at 1e-4 with weight decay 0.02, cosine schedule and 10 warmup epochs; per-GPU batch sizes are 64 for single-view/guidance and 32 for joint fusion. Those reported settings are not a substitute for the unresolved augmentation/selection details.

No original SLURM job files are available in tracked history. Preserve any recovered jobs; label newly reconstructed templates. Do not assume cluster-specific NCCL settings from a commit message apply to another machine.

## Classification and saved-score evaluation

Inspect the current interfaces without loading a model:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -B scripts_new/zero_shot_eval_vindr_pos_neg_bare_prompts.py --help
python -B scripts/evaluate_zero_shot_f1_from_validation.py --help
python -B scripts_new/learn_dual_checkpoint_ensemble_weights.py --help
```

Bare `src` imports make root `PYTHONPATH` significant for direct script execution. Package/module launch and direct execution are not generally interchangeable.

After obtaining verified score files, this is an **unexecuted template**, not a reproduced result:

```bash
python -B scripts/evaluate_zero_shot_f1_from_validation.py \
  --validation_scores /absolute/path/to/validation_scores.npz \
  --test_scores /absolute/path/to/test_scores.npz \
  --output_dir /absolute/path/to/new-run/f1 \
  --model_name verified_run_id \
  --target_label Cardiomegaly
```

Inputs require `image_ids`, `label_names`, `scores` and `y_true`. Record the actual checkpoint selection, label set and calibration. If reporting stable-label macro metrics, supply the verified label list/config explicitly; current configs and thesis differ. Per-label F1 thresholds must be fitted on validation, then frozen on test. Never overwrite an earlier result directory.

The multi-view bare-prompt evaluator currently constructs only the cross-attention architecture. Its generic filename is not support for evaluating every fusion checkpoint.

## Heatmap extraction

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -B scripts_new/extract_itc_margin_gradcam_heatmaps.py --help
```

An **unexecuted template**, after resolving the tokenizer loader and obtaining verified assets:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -B scripts_new/extract_itc_margin_gradcam_heatmaps.py \
  --config /absolute/path/to/verified-run.yaml \
  --checkpoint /absolute/path/to/selected-checkpoint.pth \
  --labels_csv /absolute/path/to/test_labels.csv \
  --images_root /absolute/path/to/test_images \
  --output_dir /absolute/path/to/new-run/maps \
  --target_labels Cardiomegaly "Pleural effusion" \
  --view_type original
```

This single-view template is appropriate only when both findings are to be explained from that same verified checkpoint. For pathology-specific guidance, run separately with its corresponding checkpoint and target label. Keep the learned temperature, penultimate visual layer, saved fields and full-image geometry. Do not enable positive-only gallery sampling for a full-cohort extraction without documenting the distinct purpose.

Fusion uses its architecture-specific extractor and branch-composite rule. BioViL-T native grounding requires its separate external environment/assets and spatial valid-region handling.

## Notebook location and working-directory context

```bash
python -B tools/notebook_context.py notebooks/evaluation/localization_pointing_game.ipynb
```

This prints canonical/legacy paths and the original file-parent directory; it does not execute cells or prove the final HPC execution directory. Open the legacy alias and verify the kernel cwd before running unchanged cells. Some older extraction notebooks unconditionally execute `os.chdir("..")`; opening a canonical deep path and running cells without checking cwd can change behavior. See the [notebook catalog](../notebooks/README.md).

## Pointing game and figures

Use [the notebook](../notebooks/evaluation/localization_pointing_game.ipynb) in a compatible analysis environment. Inspect its protocol and payloads; fill actual checkpoint provenance; confirm full-image geometry and strict coverage before scoring. Its configured sources cover guidance only; fusion requires its separate verified execution configuration and artifact bundle, not an assumption that the current notebook already reproduces Table 6.8.

Required outputs include protocol/provenance, cohort, coverage, per-image outcomes, hit-rate summary and paired comparisons. Keep missing/invalid, flat and tied maps distinguishable. Do not tune layers or thresholds against test hit rates. Keep guidance and fusion Holm families separate.

The four-radiograph qualitative figure series is distinct from full positive-box cohorts. Recover its exact case selection, map fields and annotations before exporting final figures. Patient-image figure redistribution requires review.

## Validation status

The documentation phase checked CLI/import interfaces, local links, registry paths and protected-file hashes. The structural phase additionally tests exact migrated bytes, historical/canonical launch equivalence and deterministic CPU behavior. See [migration validation](migration_validation.md). It does not validate data preparation, GPU execution, complete external-model installs, checkpoint selection or final numerical reproduction. See the [phase validation record](validation.md) for the exact commands run and their outcomes.
