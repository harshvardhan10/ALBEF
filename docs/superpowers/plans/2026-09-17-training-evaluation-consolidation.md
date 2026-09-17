# Training and Evaluation Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate duplicated multiview training, anatomy-guidance utilities, checkpoint handling, classification evaluation, and localization provenance without invalidating existing SLURM commands.

**Architecture:** Extract data, initialization, checkpoint, and evaluation responsibilities into focused modules. Use one configurable joint-multiview trainer with thin historical entry points, then share stable utilities across A0-A6 while keeping the patch-head workflow separate.

**Tech Stack:** Python 3.8.20, PyTorch 1.8.1+cu111, torchvision 0.9.1+cu111, NumPy, scikit-learn, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-09-17-thesis-repository-cleanup-design.md`

## Global Constraints

- Preserve existing SLURM command paths.
- Preserve synchronized augmentation across original, lung, and heart views.
- Preserve full-dataset configuration with `train_subset_size: null`.
- Preserve separate best-cardiomegaly and best-stable-macro checkpoints.
- Do not treat `checkpoint_last.pth` as a reported scientific checkpoint.
- Preserve validation-fitted ensemble weights and apply them unchanged to test.
- Record cardiomegaly and pleural-effusion results.
- Treat localization comparisons as qualitative.
- Keep FROC utilities exploratory.
- Support Python 3.8.20 and the successful CUDA 11.1 stack.

---

### Task 1: Extract the synchronized multiview dataset

**Files:**
- Create: `dataset/multiview_cxr_dataset.py`
- Create: `tests/dataset/test_multiview_cxr_dataset.py`
- Modify later: all six joint-multiview trainer entry points

**Interfaces:**
- Produces: `SynchronizedCXRTransform(config)`.
- Produces: `MultiViewCXRPretrainDataset(...)`.
- Produces: `build_multiview_pretrain_dataset(config) -> Dataset`.
- Returns each sample as `(original_tensor, lung_tensor, heart_tensor, caption)`.

- [ ] **Step 1: Write tests for alignment and safe paths**

```python
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from dataset.multiview_cxr_dataset import (
    MultiViewCXRPretrainDataset,
    SynchronizedCXRTransform,
)


def write_image(path, value):
    Image.new("RGB", (32, 32), (value, value, value)).save(path)


def write_mask(path):
    Image.new("L", (32, 32), 255).save(path)


def test_dataset_returns_three_aligned_views_and_caption(tmp_path):
    image = tmp_path / "image.png"
    lung = tmp_path / "lung.png"
    heart = tmp_path / "heart.png"
    write_image(image, 128)
    write_mask(lung)
    write_mask(heart)

    lung_records = [{"image": str(image), "caption": "cardiomegaly", "mask_relpath": "lung.png"}]
    heart_records = [{"image": str(image), "caption": "cardiomegaly", "mask_relpath": "heart.png"}]
    (tmp_path / "lung.json").write_text(json.dumps(lung_records))
    (tmp_path / "heart.json").write_text(json.dumps(heart_records))

    transform = SynchronizedCXRTransform(
        {"image_res": 32, "cxr_augmentation": {"enabled": False}}
    )
    dataset = MultiViewCXRPretrainDataset(
        lung_ann_files=[tmp_path / "lung.json"],
        heart_ann_files=[tmp_path / "heart.json"],
        lung_mask_root=tmp_path,
        heart_mask_root=tmp_path,
        transform=transform,
        max_words=30,
    )
    original, lung_view, heart_view, caption = dataset[0]
    assert original.shape == lung_view.shape == heart_view.shape == (3, 32, 32)
    assert caption == "cardiomegaly"


def test_dataset_rejects_manifest_misalignment(tmp_path):
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    lung_mask = tmp_path / "lung.png"
    heart_mask = tmp_path / "heart.png"
    write_image(first_image, 64)
    write_image(second_image, 192)
    write_mask(lung_mask)
    write_mask(heart_mask)

    lung_records = [
        {
            "image": str(first_image),
            "caption": "cardiomegaly",
            "mask_relpath": "lung.png",
        }
    ]
    heart_records = [
        {
            "image": str(second_image),
            "caption": "cardiomegaly",
            "mask_relpath": "heart.png",
        }
    ]
    (tmp_path / "lung.json").write_text(json.dumps(lung_records))
    (tmp_path / "heart.json").write_text(json.dumps(heart_records))

    transform = SynchronizedCXRTransform(
        {"image_res": 32, "cxr_augmentation": {"enabled": False}}
    )
    with pytest.raises(ValueError, match="misalignment"):
        MultiViewCXRPretrainDataset(
            lung_ann_files=[tmp_path / "lung.json"],
            heart_ann_files=[tmp_path / "heart.json"],
            lung_mask_root=tmp_path,
            heart_mask_root=tmp_path,
            transform=transform,
            max_words=30,
        )
```

- [ ] **Step 2: Run tests and confirm import failure**

Run:

```bash
python -m pytest tests/dataset/test_multiview_cxr_dataset.py -v
```

Expected: failure because the shared dataset module does not exist.

- [ ] **Step 3: Move the stable dataset and transform implementation**

Extract from `Pretrain_A0_multiview_fusion_dual_auc.py`:

```text
_as_path_list
load_json_records
_safe_relative_path
SynchronizedCXRTransform
MultiViewCXRPretrainDataset
build_dataset
```

Rename `build_dataset` to `build_multiview_pretrain_dataset`. Preserve synchronized geometric and photometric sampling.

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest tests/dataset/test_multiview_cxr_dataset.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add dataset/multiview_cxr_dataset.py tests/dataset/test_multiview_cxr_dataset.py
git commit -m "refactor: extract synchronized multiview dataset"
```

### Task 2: Extract multiview initialization and checkpoint loading

**Files:**
- Create: `training/__init__.py`
- Create: `training/multiview_checkpoints.py`
- Create: `tests/training/test_multiview_checkpoints.py`

**Interfaces:**
- Produces: `load_checkpoint_state(path) -> Tuple[dict, dict]`.
- Produces: `initialize_from_three_single_view_checkpoints(model, config) -> dict`.
- Produces: `load_fused_weights_only(model, path) -> LoadReport`.
- Produces: `validate_load_result(missing_keys, unexpected_keys, allowed_missing, allowed_unexpected)`.
- Requires explicit reporting of every missing and unexpected key.

- [ ] **Step 1: Write checkpoint validation tests**

```python
import pytest

from training.multiview_checkpoints import validate_load_result


def test_checkpoint_validation_accepts_only_declared_differences():
    validate_load_result(
        missing_keys=["view_fusion.weight"],
        unexpected_keys=[],
        allowed_missing={"view_fusion.weight"},
        allowed_unexpected=set(),
    )


def test_checkpoint_validation_rejects_silent_model_mismatch():
    with pytest.raises(RuntimeError, match="Unexpected checkpoint mismatch"):
        validate_load_result(
            missing_keys=["visual_encoder.blocks.0.attn.qkv.weight"],
            unexpected_keys=[],
            allowed_missing={"view_fusion.weight"},
            allowed_unexpected=set(),
        )
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/training/test_multiview_checkpoints.py -v
```

Expected: failure because the checkpoint module does not exist.

- [ ] **Step 3: Extract checkpoint helpers**

Move these functions from the stable multiview trainer:

```text
_load_checkpoint_state
_substate
_load_single_view_visual_encoder
_load_shared_from_single_view_checkpoint
initialize_from_three_single_view_checkpoints
_interpolate_fused_positional_embeddings
load_fused_weights_only
resume_checkpoint
```

Replace unchecked `strict=False` behavior with a returned report and `validate_load_result`.

- [ ] **Step 4: Add a serializable load report**

```python
from dataclasses import dataclass
from typing import List


@dataclass
class LoadReport:
    missing_keys: List[str]
    unexpected_keys: List[str]

    def to_dict(self):
        return {
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
        }
```

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest tests/training/test_multiview_checkpoints.py -v
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add training/multiview_checkpoints.py tests/training/test_multiview_checkpoints.py
git commit -m "refactor: centralize multiview checkpoint loading"
```

### Task 3: Create one configurable joint-multiview trainer

**Files:**
- Create: `training/multiview_trainer.py`
- Create: `Pretrain_A0_multiview.py`
- Create: `tests/training/test_multiview_trainer.py`
- Modify: six historical joint-fusion trainer files after the shared trainer passes tests

**Interfaces:**
- Produces: `run_multiview_training(args, config, fusion_type: str)`.
- Produces: `main_for_fusion(fusion_type: str)`.
- Consumes: shared model, dataset builder, checkpoint module, and validation runner.
- Preserves historical CLI arguments and output checkpoint names.

- [ ] **Step 1: Write configuration tests**

```python
import pytest

from training.multiview_trainer import resolve_fusion_type, validate_multiview_config


def test_explicit_fusion_type_is_recorded():
    config = {"view_fusion_type": "gated", "train_subset_size": None}
    assert resolve_fusion_type(config, None) == "gated"


def test_full_training_is_explicit():
    config = {"view_fusion_type": "mean", "train_subset_size": None}
    validate_multiview_config(config)


def test_missing_subset_field_is_rejected():
    with pytest.raises(ValueError, match="train_subset_size"):
        validate_multiview_config({"view_fusion_type": "mean"})
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/training/test_multiview_trainer.py -v
```

Expected: failure because the shared trainer does not exist.

- [ ] **Step 3: Build the shared trainer from the stable FC trainer**

Move common logic from `Pretrain_A0_multiview_fusion_dual_auc.py` into:

```python
def run_multiview_training(args, config, fusion_type):
    config = dict(config)
    config["view_fusion_type"] = fusion_type
    validate_multiview_config(config)
    dataset = build_multiview_pretrain_dataset(config)
    model = SharedMultiViewALBEF(
        config=config,
        text_encoder=config["text_encoder"],
        tokenizer=tokenizer,
        init_deit=not bool(args.checkpoint),
        fusion_type=fusion_type,
    )
    # Continue with the existing optimizer, scheduler, validation,
    # checkpoint selection, resume, and distributed training flow.
```

The implementation must copy the existing logic in full rather than replace omitted sections with new behavior.

- [ ] **Step 4: Preserve dual-checkpoint selection**

Assert these exact output names in a test using a temporary directory and a mocked validation result:

```python
assert (output_dir / "checkpoint_best_cardiomegaly_auc.pth").exists()
assert (output_dir / "checkpoint_best_macro_auc_stable.pth").exists()
```

Mock `atomic_torch_save` so the unit test does not serialize a full model.

- [ ] **Step 5: Run shared trainer tests**

Run:

```bash
python -m pytest tests/training/test_multiview_trainer.py -v
```

Expected: pass.

- [ ] **Step 6: Commit the shared trainer**

```bash
git add training/multiview_trainer.py Pretrain_A0_multiview.py tests/training/test_multiview_trainer.py
git commit -m "refactor: add configurable multiview trainer"
```

### Task 4: Convert historical fusion trainers into CLI wrappers

**Files:**
- Modify: `Pretrain_A0_multiview_fusion_dual_auc.py`
- Modify: `Pretrain_A0_multiview_mean_fusion_dual_auc.py`
- Modify: `Pretrain_A0_multiview_gated_fusion_dual_auc.py`
- Modify: `Pretrain_A0_multiview_residual_fusion_dual_auc.py`
- Modify: `Pretrain_A0_multiview_transformer_fusion_dual_auc.py`
- Modify: `Pretrain_A0_multiview_cross_attention_fusion_dual_auc.py`
- Create: `tests/training/test_training_wrappers.py`

**Interfaces:**
- Produces: unchanged script filenames for SLURM.
- Delegates: `main_for_fusion("<fusion_type>")`.

- [ ] **Step 1: Write wrapper-size and delegation tests**

```python
from pathlib import Path
import pytest

CASES = {
    "Pretrain_A0_multiview_fusion_dual_auc.py": "fc",
    "Pretrain_A0_multiview_mean_fusion_dual_auc.py": "mean",
    "Pretrain_A0_multiview_gated_fusion_dual_auc.py": "gated",
    "Pretrain_A0_multiview_residual_fusion_dual_auc.py": "residual",
    "Pretrain_A0_multiview_transformer_fusion_dual_auc.py": "transformer",
    "Pretrain_A0_multiview_cross_attention_fusion_dual_auc.py": "cross_attention",
}


@pytest.mark.parametrize("path,fusion_type", CASES.items())
def test_wrapper_delegates_to_shared_trainer(path, fusion_type):
    text = Path(path).read_text(encoding="utf-8")
    assert "main_for_fusion" in text
    assert repr(fusion_type) in text or f'"{fusion_type}"' in text
    assert len(text.splitlines()) <= 40
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/training/test_training_wrappers.py -v
```

Expected: failure because the historical scripts still contain full trainer copies.

- [ ] **Step 3: Replace each script with a thin wrapper**

```python
from training.multiview_trainer import main_for_fusion


if __name__ == "__main__":
    main_for_fusion("gated")
```

Use the corresponding fusion type in each file. Preserve executable entry behavior.

- [ ] **Step 4: Run wrapper and trainer tests**

Run:

```bash
python -m pytest tests/training/test_training_wrappers.py tests/training/test_multiview_trainer.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Pretrain_A0_multiview_* training tests/training
git commit -m "refactor: replace fusion trainers with compatibility wrappers"
```

### Task 5: Make experiment configurations immutable

**Files:**
- Create: `configs/experiments/stage_04_single_view/`
- Create: `configs/experiments/stage_05_multiview/`
- Create: `configs/environments/cluster.example.yaml`
- Create: `tests/configs/test_experiment_configs.py`
- Retain: old configuration paths as copies or redirect documentation until SLURM scripts migrate

**Interfaces:**
- Produces: one immutable YAML file per completed experiment.
- Requires: `train_subset_size: null` in full-dataset experiments.
- Requires: no `/home/woody` or other personal absolute paths in experiment configs.

- [ ] **Step 1: Write configuration tests**

```python
from pathlib import Path
import yaml

ROOT = Path("configs/experiments")


def experiment_configs():
    return sorted(ROOT.rglob("*.yaml"))


def test_experiment_configs_are_full_dataset_and_portable():
    assert experiment_configs()
    for path in experiment_configs():
        text = path.read_text(encoding="utf-8")
        config = yaml.safe_load(text)
        assert "train_subset_size" in config
        assert config["train_subset_size"] is None
        assert "/home/" not in text
        assert "view_fusion_type" in config or "view_type" in config
```

Split the final assertion by stage if anatomy configs are added in a later task.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/configs/test_experiment_configs.py -v
```

Expected: failure because immutable experiment directories do not exist.

- [ ] **Step 3: Create Stage 4 configs**

Create explicit full-dataset configs for:

```text
original.yaml
lung.yaml
heart.yaml
```

Each config must identify `view_type`, validation split, label list, and both checkpoint-selection roles.

- [ ] **Step 4: Create Stage 5 configs**

Create explicit configs for:

```text
fc.yaml
mean.yaml
gated.yaml
residual.yaml
transformer.yaml
cross_attention.yaml
```

Every file must contain `view_fusion_type`, `train_subset_size: null`, the relevant fusion block, and portable data keys.

- [ ] **Step 5: Create the environment example**

```yaml
mimic_manifest: /path/to/mimic_cxr.json
lung_manifest: /path/to/mimic_lung_view.json
heart_manifest: /path/to/mimic_heart_view.json
lung_mask_root: /path/to/lung_masks
heart_mask_root: /path/to/heart_masks
vindr_image_root: /path/to/vindr/images
vindr_validation_csv: /path/to/vindr/validation.csv
```

Training code must merge the selected experiment config with an environment config or equivalent explicit CLI values and save the resolved configuration in the output directory.

- [ ] **Step 6: Run config tests**

Run:

```bash
python -m pytest tests/configs/test_experiment_configs.py -v
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add configs/experiments configs/environments tests/configs
git commit -m "config: add immutable single-view and multiview experiments"
```

### Task 6: Consolidate A0-A4 anatomy-guidance utilities

**Files:**
- Create: `training/anatomy_guidance.py`
- Modify: `Pretrain_anatomy_prior.py`
- Modify: `Pretrain_anatomy_prior_A4.py`
- Create: `configs/experiments/stage_02_anatomy_guidance/`
- Create: `tests/training/test_anatomy_guidance.py`

**Interfaces:**
- Produces: `build_support_weights_from_captions(text, config, device)`.
- Produces: prior-loading and prior-mask functions.
- Produces: classification-preservation utilities used only when enabled.
- Preserves: A0-A3 standard anatomy trainer and A4 teacher-based preservation behavior.

- [ ] **Step 1: Write support-weight tests**

```python
import torch

from training.anatomy_guidance import build_support_weights_from_captions


def test_positive_only_support_weights():
    weights = build_support_weights_from_captions(
        ["cardiomegaly", "uncertain cardiomegaly", "no finding"],
        {
            "support_mode": "positive_only",
            "anatomy_target_phrase": "cardiomegaly",
        },
        torch.device("cpu"),
    )
    torch.testing.assert_close(weights, torch.tensor([1.0, 0.0, 0.0]))


def test_uncertainty_weighted_support_weights():
    weights = build_support_weights_from_captions(
        ["cardiomegaly", "uncertain cardiomegaly", "no finding"],
        {
            "support_mode": "uncertainty_weighted",
            "anatomy_target_phrase": "cardiomegaly",
            "positive_caption_weight": 1.0,
            "uncertain_caption_weight": 0.5,
        },
        torch.device("cpu"),
    )
    torch.testing.assert_close(weights, torch.tensor([1.0, 0.5, 0.0]))
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/training/test_anatomy_guidance.py -v
```

Expected: failure because the shared module does not exist.

- [ ] **Step 3: Extract identical helpers**

Move support weighting and anatomy-prior construction into `training/anatomy_guidance.py`. Keep the compatibility alias `all_cardiomegaly_captions -> all_target_captions`.

Move A4 classification-preservation helpers into the same module under explicit names:

```python
classification_preservation_loss
get_global_image_text_features
load_teacher_checkpoint
```

- [ ] **Step 4: Add immutable A0-A4 configs**

Each config must set `train_subset_size: null`, support mode, anatomy target phrase, loss coefficient, and A4 preservation settings where applicable. Machine paths belong in the environment config.

- [ ] **Step 5: Update trainers to import shared helpers**

Do not combine the A4 teacher training loop with A0-A3 until shared-helper equivalence passes. Remove only duplicated helper bodies in this task.

- [ ] **Step 6: Run tests and syntax checks**

Run:

```bash
python -m pytest tests/training/test_anatomy_guidance.py -v
python -m py_compile Pretrain_anatomy_prior.py Pretrain_anatomy_prior_A4.py training/anatomy_guidance.py
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add training/anatomy_guidance.py Pretrain_anatomy_prior.py Pretrain_anatomy_prior_A4.py configs/experiments/stage_02_anatomy_guidance tests/training/test_anatomy_guidance.py
git commit -m "refactor: share A0 to A4 anatomy guidance utilities"
```

### Task 7: Consolidate patch-refinement utilities while preserving separate workflows

**Files:**
- Create: `training/patch_refinement.py`
- Modify: `Pretrain_anatomy_prior_A5_patch_head.py`
- Modify: `Pretrain_anatomy_prior_A6_1_frozen_A0_patch_head.py`
- Create: `configs/experiments/stage_03_patch_refinement/`
- Create: `tests/training/test_patch_refinement.py`

**Interfaces:**
- Produces: shared checkpoint extraction, weighted mean, gradient synchronization, norm reporting, and identity loss.
- Preserves: A5 trainable ALBEF behavior and A6.1 frozen ALBEF behavior as separate entry points.

- [ ] **Step 1: Write identity-loss tests**

```python
import torch

from training.patch_refinement import patch_identity_loss


def test_identity_loss_is_zero_for_equal_maps():
    maps = torch.tensor([[0.2, 0.8]])
    weights = torch.tensor([1.0])
    loss = patch_identity_loss(maps, maps.clone(), weights, mode="mse")
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_identity_loss_ignores_inactive_samples():
    prediction = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    weights = torch.tensor([0.0, 1.0])
    loss = patch_identity_loss(prediction, target, weights, mode="mse")
    torch.testing.assert_close(loss, torch.tensor(0.0))
```

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
python -m pytest tests/training/test_patch_refinement.py -v
```

Expected: failure because the shared patch module does not exist.

- [ ] **Step 3: Extract shared utility functions**

Move and unify:

```text
get_model_without_ddp
weighted_mean
module_grad_norm
module_param_norm
sync_module_gradients
load_model_albef_state
extract_model_state_from_checkpoint
patch_identity_loss
```

Keep checkpoint payload keys `model` and `patch_head`.

- [ ] **Step 4: Add immutable A5 and A6.1 configs**

Both configs use `train_subset_size: null`. A6.1 explicitly enables frozen ALBEF. A5 explicitly enables normal ALBEF objectives. Neither config is described as semi-supervised.

- [ ] **Step 5: Run tests and compile both trainers**

Run:

```bash
python -m pytest tests/training/test_patch_refinement.py -v
python -m py_compile Pretrain_anatomy_prior_A5_patch_head.py Pretrain_anatomy_prior_A6_1_frozen_A0_patch_head.py training/patch_refinement.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add training/patch_refinement.py Pretrain_anatomy_prior_A5_patch_head.py Pretrain_anatomy_prior_A6_1_frozen_A0_patch_head.py configs/experiments/stage_03_patch_refinement tests/training/test_patch_refinement.py
git commit -m "refactor: share patch refinement utilities"
```

### Task 8: Centralize evaluation metadata and dual-checkpoint reporting

**Files:**
- Create: `evaluation/__init__.py`
- Create: `evaluation/experiment_metadata.py`
- Create: `evaluation/checkpoint_roles.py`
- Modify: `scripts/vindr_classification_validation.py`
- Modify: `scripts_new/vindr_multiview_classification_validation.py`
- Modify: `scripts_new/learn_dual_checkpoint_ensemble_weights.py`
- Create: `tests/evaluation/test_experiment_metadata.py`

**Interfaces:**
- Produces: canonical label order and prompt metadata.
- Produces: `CheckpointRole` values `best_cardiomegaly_auc` and `best_macro_auc_stable`.
- Produces: a JSON-serializable evaluation record.

- [ ] **Step 1: Write metadata tests**

```python
from evaluation.experiment_metadata import EvaluationRecord


def test_evaluation_record_requires_provenance():
    record = EvaluationRecord(
        experiment_id="stage5-fc",
        git_commit="0123456789abcdef",
        checkpoint_role="best_cardiomegaly_auc",
        split="vindr_test",
        localization_method=None,
        classification_metrics={"cardiomegaly_auc": 0.9},
    )
    payload = record.to_dict()
    assert payload["checkpoint_role"] == "best_cardiomegaly_auc"
    assert payload["split"] == "vindr_test"
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/evaluation/test_experiment_metadata.py -v
```

Expected: failure because the evaluation package does not exist.

- [ ] **Step 3: Implement immutable evaluation records**

Use a frozen dataclass with required experiment ID, commit, checkpoint role, split, optional localization method, and metric dictionary. Validate checkpoint roles against the two scientific roles.

- [ ] **Step 4: Update classification runners**

Write the resolved evaluation metadata next to every NPZ or JSON result. Preserve current label alignment and validation-selected thresholds.

- [ ] **Step 5: Update ensemble output**

The learned ensemble must save:

- validation-fitted normalization parameters;
- validation-fitted simplex weights;
- label order;
- input checkpoint roles;
- validation metrics;
- test metrics produced without refitting.

- [ ] **Step 6: Run tests**

Run:

```bash
python -m pytest tests/evaluation/test_experiment_metadata.py -v
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add evaluation scripts/vindr_classification_validation.py scripts_new/vindr_multiview_classification_validation.py scripts_new/learn_dual_checkpoint_ensemble_weights.py tests/evaluation
git commit -m "refactor: record evaluation and checkpoint provenance"
```

### Task 9: Catalogue localization methods and isolate exploratory FROC code

**Files:**
- Create: `localization/methods.py`
- Create: `docs/localization_methods.md`
- Create: `scripts/exploratory_froc/README.md`
- Move: FROC scripts into `scripts/exploratory_froc/` with compatibility wrappers at old paths
- Create: `tests/localization/test_method_metadata.py`

**Interfaces:**
- Produces: `LocalizationMethod` metadata with method, score target, attention source, layer, view, head aggregation, ReLU placement, normalization, and interpolation.
- Preserves: old FROC script command paths through wrappers.
- Makes no quantitative localization claim.

- [ ] **Step 1: Write localization metadata tests**

```python
from localization.methods import LocalizationMethod


def test_localization_method_serializes_methodological_choices():
    method = LocalizationMethod(
        name="itc_margin_gradcam",
        score_target="positive_minus_negative_itc",
        attention_source="vit_self_attention",
        layer=10,
        view="original",
        head_aggregation="mean_after_per_head_relu",
        normalization="per_map_minmax",
        interpolation="bilinear",
    )
    payload = method.to_dict()
    assert payload["score_target"] == "positive_minus_negative_itc"
    assert payload["head_aggregation"] == "mean_after_per_head_relu"
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/localization/test_method_metadata.py -v
```

Expected: failure because localization metadata is not centralized.

- [ ] **Step 3: Implement the frozen metadata class**

Validate non-empty method name, score target, attention source, aggregation, normalization, and interpolation. Allow `layer=None` only for native phrase grounding or learned patch heads.

- [ ] **Step 4: Document every used localization method**

Include ALBEF ITC Grad-CAM, ITC-margin Grad-CAM, cross-attention Grad-CAM, learned patch head, BioViL-T native phrase grounding, CheXzero attention Grad-CAM, and heatmap ensembles.

- [ ] **Step 5: Isolate FROC scripts**

Move implementations into `scripts/exploratory_froc/`. Leave old filenames as wrappers that emit a clear exploratory-status message and delegate to the moved implementation.

- [ ] **Step 6: Run localization tests and compile wrappers**

Run:

```bash
python -m pytest tests/localization/test_method_metadata.py -v
python -m compileall localization scripts/exploratory_froc
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add localization docs/localization_methods.md scripts tests/localization
git commit -m "refactor: document localization methods and isolate FROC utilities"
```

### Task 10: Add cluster smoke-test instructions and run the local verification suite

**Files:**
- Create: `docs/cluster_smoke_tests.md`
- Modify: `docs/reproduction.md`
- Modify: `docs/experiment_catalogue.md`

**Interfaces:**
- Consumes: shared trainers, immutable configs, companion SLURM repository.
- Produces: commands for one-batch import, forward, checkpoint-load, and validation checks on the cluster.

- [ ] **Step 1: Document environment verification**

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
assert torch.__version__ == "1.8.1+cu111"
assert torch.version.cuda == "11.1"
PY
```

- [ ] **Step 2: Document one smoke command per workflow**

Provide commands for:

- A0-A4 anatomy guidance;
- A5 patch refinement;
- A6.1 frozen patch refinement;
- original/lung/heart single-view training;
- each joint fusion type;
- dual-checkpoint VinDr evaluation.

Every smoke command must use a debug config that processes at most two batches and writes to a disposable output directory.

- [ ] **Step 3: Run local tests**

Run:

```bash
python -m pytest tests -v
python -m compileall models dataset training evaluation localization scripts scripts_new
```

Expected: all tests pass and compileall exits zero.

- [ ] **Step 4: Record local limitations**

State that passing local tests proves configuration, import, shape, metadata, and extracted-module equivalence only. It does not prove full distributed-training equivalence.

- [ ] **Step 5: Commit**

```bash
git add docs/cluster_smoke_tests.md docs/reproduction.md docs/experiment_catalogue.md
git commit -m "docs: add cluster smoke-test protocol"
```
