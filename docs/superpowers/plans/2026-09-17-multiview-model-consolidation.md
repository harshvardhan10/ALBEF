# Multiview Model Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace six duplicated multiview ALBEF implementations with one shared model and six independently reviewable fusion modules while preserving checkpoint keys and numerical behavior.

**Architecture:** Keep the existing root `models` package during this pass so cluster imports remain valid. Extract fusion behavior into `models/fusion`, parameterize one shared `models/model_pretrain_multiview.py`, and reduce historical model files to thin compatibility subclasses that select one fusion type.

**Tech Stack:** Python 3.8.20, PyTorch 1.8.1+cu111, pytest

**Spec:** `docs/superpowers/specs/2026-09-17-thesis-repository-cleanup-design.md`

## Global Constraints

- Preserve the public constructor name `ALBEF`.
- Preserve state-dict prefixes `view_fusion.*` and `view_fusion_m.*`.
- Preserve separate online and momentum fusion modules.
- Preserve FC equal-average initialization.
- Preserve gated sample-specific weights.
- Preserve original-anchored residual behavior.
- Preserve Transformer two-layer token-position behavior.
- Preserve the currently running cross-attention implementation exactly before any cleanup.
- Keep all six legacy model import paths working.
- Support Python 3.8.20 and PyTorch 1.8.1.
- Do not change ITC, ITM, MLM, queue, momentum, or negative-sampling behavior.

---

### Task 1: Define the common fusion contract and extract FC and mean fusion

**Files:**
- Create: `models/fusion/__init__.py`
- Create: `models/fusion/factory.py`
- Create: `models/fusion/fc.py`
- Create: `models/fusion/mean.py`
- Create: `tests/models/test_fusion_fc_mean.py`

**Interfaces:**
- Produces: `build_fusion_module(name: str, hidden_dim: int, config: dict) -> torch.nn.Module`.
- Produces: fusion modules with `forward(original_tokens, lung_tokens, heart_tokens) -> torch.Tensor`.
- Requires: each input has identical `[batch, tokens, hidden_dim]` shape.
- Raises: `ValueError` for shape mismatch or unknown fusion name.

- [ ] **Step 1: Write failing tests for the shared contract**

```python
import torch
import pytest

from models.fusion.factory import build_fusion_module


def three_views():
    torch.manual_seed(7)
    return tuple(torch.randn(2, 5, 8) for _ in range(3))


def test_mean_fusion_is_exact_tokenwise_average():
    original, lung, heart = three_views()
    module = build_fusion_module("mean", 8, {})
    actual = module(original, lung, heart)
    expected = (original + lung + heart) / 3.0
    torch.testing.assert_close(actual, expected)


def test_fc_starts_as_equal_average_followed_by_layer_norm():
    original, lung, heart = three_views()
    module = build_fusion_module("fc", 8, {})
    actual = module(original, lung, heart)
    expected = module.output_norm((original + lung + heart) / 3.0)
    torch.testing.assert_close(actual, expected)


def test_fusion_rejects_misaligned_views():
    module = build_fusion_module("mean", 8, {})
    with pytest.raises(ValueError, match="identical"):
        module(torch.randn(2, 5, 8), torch.randn(2, 4, 8), torch.randn(2, 5, 8))
```

- [ ] **Step 2: Run tests and confirm import failure**

Run:

```bash
python -m pytest tests/models/test_fusion_fc_mean.py -v
```

Expected: failure because `models.fusion` does not exist.

- [ ] **Step 3: Implement shared input validation**

```python
def validate_view_tokens(original_tokens, lung_tokens, heart_tokens):
    shapes = [tuple(x.shape) for x in (original_tokens, lung_tokens, heart_tokens)]
    if len(set(shapes)) != 1:
        raise ValueError("All view token tensors must have identical shapes")
    if original_tokens.ndim != 3:
        raise ValueError("View token tensors must have shape [batch, tokens, hidden]")
```

- [ ] **Step 4: Implement FC and mean modules**

```python
class MeanFusion(nn.Module):
    def forward(self, original_tokens, lung_tokens, heart_tokens):
        validate_view_tokens(original_tokens, lung_tokens, heart_tokens)
        return (original_tokens + lung_tokens + heart_tokens) / 3.0


class FullyConnectedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(3 * hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.bias.zero_()
            eye = torch.eye(self.projection.out_features)
            width = self.projection.out_features
            for index in range(3):
                self.projection.weight[:, index * width:(index + 1) * width].copy_(eye / 3.0)
            self.output_norm.weight.fill_(1.0)
            self.output_norm.bias.zero_()

    def forward(self, original_tokens, lung_tokens, heart_tokens):
        validate_view_tokens(original_tokens, lung_tokens, heart_tokens)
        concatenated = torch.cat([original_tokens, lung_tokens, heart_tokens], dim=-1)
        return self.output_norm(self.projection(concatenated))
```

Use attribute names in the shared model so final state keys remain `view_fusion.*`; when mapping the historical FC checkpoint, translate `view_fusion.weight` to `view_fusion.projection.weight` only if the module extraction makes that unavoidable. Prefer defining FC parameters directly as `weight` and `bias` or providing a deterministic compatibility mapper so the historical keys remain loadable.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest tests/models/test_fusion_fc_mean.py -v
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add models/fusion tests/models/test_fusion_fc_mean.py
git commit -m "refactor: extract FC and mean fusion modules"
```

### Task 2: Extract gated and residual fusion with legacy equivalence tests

**Files:**
- Create: `models/fusion/gated.py`
- Create: `models/fusion/residual.py`
- Modify: `models/fusion/factory.py`
- Create: `tests/models/test_fusion_gated_residual.py`

**Interfaces:**
- Consumes: the common three-view token contract.
- Produces: `GatedFusion.compute_weights(...) -> Tensor[batch, 3]`.
- Produces: `ResidualFusion.compute_gates(...) -> Tensor[batch, 2]`.
- Preserves: legacy class parameter names and initialization values.

- [ ] **Step 1: Write tests against the current legacy classes**

```python
import torch

from models.fusion.factory import build_fusion_module
from models.model_pretrain_multiview_gated_fusion import ImageLevelGatedFusion
from models.model_pretrain_multiview_residual_fusion import OriginalAnchoredResidualFusion


def test_gated_matches_legacy_after_state_transfer():
    torch.manual_seed(11)
    legacy = ImageLevelGatedFusion(hidden_dim=8, gate_hidden_dim=8)
    extracted = build_fusion_module(
        "gated", 8, {"view_fusion_gate": {"hidden_dim": 8}}
    )
    extracted.load_state_dict(legacy.state_dict())
    views = tuple(torch.randn(3, 5, 8) for _ in range(3))
    torch.testing.assert_close(extracted(*views), legacy(*views))
    weights = extracted.compute_weights(*views)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))


def test_residual_matches_legacy_after_state_transfer():
    torch.manual_seed(13)
    legacy = OriginalAnchoredResidualFusion(hidden_dim=8)
    extracted = build_fusion_module("residual", 8, {})
    extracted.load_state_dict(legacy.state_dict())
    views = tuple(torch.randn(3, 5, 8) for _ in range(3))
    torch.testing.assert_close(extracted(*views), legacy(*views))
```

Match constructor arguments to the existing class definitions exactly when implementing the tests.

- [ ] **Step 2: Run tests and confirm missing factory variants**

Run:

```bash
python -m pytest tests/models/test_fusion_gated_residual.py -v
```

Expected: failure because the factory does not yet implement `gated` and `residual`.

- [ ] **Step 3: Move the legacy fusion classes without changing their internals**

Copy the class bodies from:

```text
models/model_pretrain_multiview_gated_fusion.py:21-88
models/model_pretrain_multiview_residual_fusion.py:21-103
```

Rename classes only through aliases in `models/fusion/__init__.py` if necessary. Do not change activation functions, CLS-based gate inputs, normalization, gate bias, or return behavior.

- [ ] **Step 4: Extend the factory**

```python
if name == "gated":
    gate_config = config.get("view_fusion_gate", {})
    return GatedFusion(
        hidden_dim=hidden_dim,
        gate_hidden_dim=int(gate_config.get("hidden_dim", hidden_dim)),
    )
if name == "residual":
    residual_config = config.get("view_fusion_residual", {})
    return ResidualFusion(
        hidden_dim=hidden_dim,
        gate_hidden_dim=int(residual_config.get("gate_hidden_dim", hidden_dim)),
        gate_init_bias=float(residual_config.get("gate_init_bias", -2.0)),
    )
```

Use the exact existing config key names from the legacy constructors.

- [ ] **Step 5: Run equivalence tests**

Run:

```bash
python -m pytest tests/models/test_fusion_gated_residual.py -v
```

Expected: pass with elementwise equality within PyTorch default tolerances.

- [ ] **Step 6: Commit**

```bash
git add models/fusion tests/models/test_fusion_gated_residual.py
git commit -m "refactor: extract gated and residual fusion modules"
```

### Task 3: Extract Transformer fusion without behavioral changes

**Files:**
- Create: `models/fusion/transformer.py`
- Modify: `models/fusion/factory.py`
- Create: `tests/models/test_fusion_transformer.py`

**Interfaces:**
- Produces: `TransformerFusion.forward(...) -> Tensor[batch, tokens, hidden_dim]`.
- Preserves: token-position-wise three-view Transformer processing and final mean pooling.

- [ ] **Step 1: Write a legacy equivalence test**

```python
import torch

from models.fusion.factory import build_fusion_module
from models.model_pretrain_multiview_transformer_fusion import ViewTransformerFusion


def test_transformer_matches_legacy_in_eval_mode():
    torch.manual_seed(17)
    legacy = ViewTransformerFusion(
        hidden_dim=8,
        num_heads=2,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
    ).eval()
    extracted = build_fusion_module(
        "transformer",
        8,
        {
            "view_fusion_transformer": {
                "num_heads": 2,
                "num_layers": 2,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
            }
        },
    ).eval()
    extracted.load_state_dict(legacy.state_dict())
    views = tuple(torch.randn(2, 5, 8) for _ in range(3))
    torch.testing.assert_close(extracted(*views), legacy(*views))
```

- [ ] **Step 2: Run the test and confirm failure**

Run:

```bash
python -m pytest tests/models/test_fusion_transformer.py -v
```

Expected: failure because the factory does not implement Transformer fusion.

- [ ] **Step 3: Copy the existing Transformer fusion implementation**

Move the class body from:

```text
models/model_pretrain_multiview_transformer_fusion.py:35-148
```

Preserve view embeddings, Transformer layer order, reshape order, mean pooling, and output normalization.

- [ ] **Step 4: Run the equivalence test**

Run:

```bash
python -m pytest tests/models/test_fusion_transformer.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add models/fusion/transformer.py models/fusion/factory.py tests/models/test_fusion_transformer.py
git commit -m "refactor: extract Transformer fusion module"
```

### Task 4: Snapshot and extract the ongoing cross-attention fusion

**Files:**
- Create: `models/fusion/cross_attention.py`
- Modify: `models/fusion/factory.py`
- Create: `tests/models/test_fusion_cross_attention.py`
- Create: `docs/cross_attention_provenance.md`

**Interfaces:**
- Produces: anchored cross-attention fusion using original tokens as the default anchor and lung/heart as ordered memory views.
- Preserves: no mean pooling; the twice-updated anchor query is the fused token.

- [ ] **Step 1: Record the source implementation**

Document:

```text
Source branch: main at cleanup start
Source tree commit: 4eabfaebe768533d85d55e192f42322c4ab67061
Source file: models/model_pretrain_multiview_cross_attention_fusion.py
Experiment status: ongoing
Default anchor: original
Default memory order: lung, heart
```

Before extraction, confirm the active cluster job was launched from this commit. If the launched commit differs, record that exact commit and use its file as the source.

- [ ] **Step 2: Write an equivalence test**

```python
import torch

from models.fusion.factory import build_fusion_module
from models.model_pretrain_multiview_cross_attention_fusion import (
    ViewCrossAttentionFusion,
)


def test_cross_attention_matches_running_implementation():
    torch.manual_seed(19)
    kwargs = dict(
        hidden_dim=8,
        num_heads=2,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        anchor_view="original",
        memory_views=("lung", "heart"),
    )
    legacy = ViewCrossAttentionFusion(**kwargs).eval()
    extracted = build_fusion_module(
        "cross_attention",
        8,
        {
            "view_fusion_cross_attention": {
                "num_heads": 2,
                "num_layers": 2,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
                "anchor_view": "original",
                "memory_views": ["lung", "heart"],
            }
        },
    ).eval()
    extracted.load_state_dict(legacy.state_dict())
    views = tuple(torch.randn(2, 5, 8) for _ in range(3))
    torch.testing.assert_close(extracted(*views), legacy(*views))
```

Adapt the constructor argument order to the source class signature without changing values.

- [ ] **Step 3: Run the test and confirm failure**

Run:

```bash
python -m pytest tests/models/test_fusion_cross_attention.py -v
```

Expected: failure because the extracted implementation does not exist.

- [ ] **Step 4: Copy the current cross-attention classes exactly**

Move the class bodies from:

```text
models/model_pretrain_multiview_cross_attention_fusion.py:41-252
```

Do not simplify the two anchored layers, change residual order, alter normalization, reorder memory views, or add mean pooling.

- [ ] **Step 5: Run the equivalence test**

Run:

```bash
python -m pytest tests/models/test_fusion_cross_attention.py -v
```

Expected: pass.

- [ ] **Step 6: Commit the provenance-locked extraction**

```bash
git add models/fusion/cross_attention.py models/fusion/factory.py tests/models/test_fusion_cross_attention.py docs/cross_attention_provenance.md
git commit -m "refactor: extract provenance-locked cross-attention fusion"
```

### Task 5: Create the shared multiview ALBEF model

**Files:**
- Create: `models/model_pretrain_multiview.py`
- Create: `tests/models/test_multiview_model_structure.py`

**Interfaces:**
- Produces: `ALBEF(config, text_encoder=None, tokenizer=None, init_deit=True, fusion_type=None)`.
- Consumes: `build_fusion_module(fusion_type, vision_width, config)`.
- Preserves: `view_fusion`, `view_fusion_m`, `model_pairs`, `encode_image_views`, `get_image_features`, and the existing `forward` return values.

- [ ] **Step 1: Write structural tests**

```python
from pathlib import Path


def test_shared_model_contains_one_training_objective_implementation():
    text = Path("models/model_pretrain_multiview.py").read_text(encoding="utf-8")
    assert text.count("def forward(") == 1
    assert "build_fusion_module" in text
    assert "self.view_fusion" in text
    assert "self.view_fusion_m" in text
    assert "self._momentum_update()" in text


def test_shared_model_keeps_expected_public_methods():
    text = Path("models/model_pretrain_multiview.py").read_text(encoding="utf-8")
    for signature in [
        "def encode_image_views(",
        "def get_image_features(",
        "def copy_params(",
        "def reset_momentum_from_online(",
        "def _momentum_update(",
        "def _dequeue_and_enqueue(",
    ]:
        assert signature in text
```

- [ ] **Step 2: Run the test and confirm failure**

Run:

```bash
python -m pytest tests/models/test_multiview_model_structure.py -v
```

Expected: failure because the shared model does not exist.

- [ ] **Step 3: Build the shared model from the mean-fusion implementation**

Use `models/model_pretrain_multiview_mean_fusion.py` as the initial shared body because its fusion call is already module-based and its remaining ALBEF logic matches the other stable fusion models.

Replace hard-coded construction with:

```python
selected_fusion = fusion_type or config.get("view_fusion_type")
if not selected_fusion:
    raise ValueError("view_fusion_type must identify a registered fusion module")

self.view_fusion = build_fusion_module(selected_fusion, vision_width, config)
self.view_fusion_m = build_fusion_module(selected_fusion, vision_width, config)
```

Keep both modules in `model_pairs`. Keep all non-fusion methods byte-for-byte equivalent to the selected stable source until tests pass.

- [ ] **Step 4: Add configuration validation**

Reject unknown fusion types before constructing encoders. Check that hidden size, head count, and view ordering are valid through the fusion factory.

- [ ] **Step 5: Run model and fusion tests**

Run:

```bash
python -m pytest tests/models/test_fusion_fc_mean.py tests/models/test_fusion_gated_residual.py tests/models/test_fusion_transformer.py tests/models/test_fusion_cross_attention.py tests/models/test_multiview_model_structure.py -v
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add models/model_pretrain_multiview.py tests/models/test_multiview_model_structure.py
git commit -m "refactor: add shared multiview ALBEF model"
```

### Task 6: Replace duplicated model files with compatibility wrappers

**Files:**
- Modify: `models/model_pretrain_multiview_fusion.py`
- Modify: `models/model_pretrain_multiview_mean_fusion.py`
- Modify: `models/model_pretrain_multiview_gated_fusion.py`
- Modify: `models/model_pretrain_multiview_residual_fusion.py`
- Modify: `models/model_pretrain_multiview_transformer_fusion.py`
- Modify: `models/model_pretrain_multiview_cross_attention_fusion.py`
- Create: `tests/models/test_multiview_compatibility_wrappers.py`

**Interfaces:**
- Consumes: shared `models.model_pretrain_multiview.ALBEF`.
- Produces: unchanged historical import paths and the class name `ALBEF`.

- [ ] **Step 1: Write wrapper tests**

```python
import importlib
import inspect
import pytest

CASES = [
    ("models.model_pretrain_multiview_fusion", "fc"),
    ("models.model_pretrain_multiview_mean_fusion", "mean"),
    ("models.model_pretrain_multiview_gated_fusion", "gated"),
    ("models.model_pretrain_multiview_residual_fusion", "residual"),
    ("models.model_pretrain_multiview_transformer_fusion", "transformer"),
    ("models.model_pretrain_multiview_cross_attention_fusion", "cross_attention"),
]


@pytest.mark.parametrize("module_name,fusion_type", CASES)
def test_legacy_module_is_a_thin_shared_model_wrapper(module_name, fusion_type):
    module = importlib.import_module(module_name)
    source = inspect.getsource(module.ALBEF)
    assert fusion_type in source
    assert len(source.splitlines()) <= 20
```

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
python -m pytest tests/models/test_multiview_compatibility_wrappers.py -v
```

Expected: failure because each file still contains a complete model copy.

- [ ] **Step 3: Replace each model with a thin subclass**

Use this pattern:

```python
from models.model_pretrain_multiview import ALBEF as SharedMultiViewALBEF


class ALBEF(SharedMultiViewALBEF):
    def __init__(self, config, text_encoder=None, tokenizer=None, init_deit=True):
        super().__init__(
            config=config,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            init_deit=init_deit,
            fusion_type="mean",
        )
```

Change only the final fusion type in each wrapper.

Re-export historical fusion class names from `models.fusion` when evaluation or extraction scripts import them directly.

- [ ] **Step 4: Verify state-dict compatibility**

Create one shared-model instance per fusion type using the smallest test config that can construct the modules. Assert online and momentum keys begin with the same prefixes expected by historical checkpoints:

```python
keys = model.state_dict().keys()
assert any(key.startswith("view_fusion.") for key in keys) or fusion_type == "mean"
assert any(key.startswith("view_fusion_m.") for key in keys) or fusion_type == "mean"
```

For FC, add and test a deterministic key conversion only if parameter extraction changed historical names.

- [ ] **Step 5: Run the complete model test suite**

Run:

```bash
python -m pytest tests/models -v
```

Expected: pass.

- [ ] **Step 6: Commit the duplicate removal**

```bash
git add models tests/models
git commit -m "refactor: replace multiview model copies with wrappers"
```

### Task 7: Verify model-source reduction and document architecture

**Files:**
- Create: `docs/model_architecture.md`
- Modify: `docs/experiment_catalogue.md`
- Modify: `README.md`
- Create: `tests/models/test_model_documentation.py`

**Interfaces:**
- Consumes: shared model and six fusion modules.
- Produces: supervisor-facing model map and duplication guard.

- [ ] **Step 1: Add a duplication guard**

```python
from pathlib import Path


def test_legacy_multiview_model_wrappers_are_small():
    paths = [
        "models/model_pretrain_multiview_fusion.py",
        "models/model_pretrain_multiview_mean_fusion.py",
        "models/model_pretrain_multiview_gated_fusion.py",
        "models/model_pretrain_multiview_residual_fusion.py",
        "models/model_pretrain_multiview_transformer_fusion.py",
        "models/model_pretrain_multiview_cross_attention_fusion.py",
    ]
    for path in paths:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 40, f"{path} is no longer a thin wrapper"
```

- [ ] **Step 2: Document the shared data flow**

`docs/model_architecture.md` must explain:

```text
three view-specific ViTs
    -> aligned token sequences
    -> selected fusion module
    -> one fused token sequence
    -> shared ITC, ITM, and MLM objectives
    -> mirrored momentum fusion
    -> one fused image feature in the queue
```

Add a table comparing parameters, initialization, anchor behavior, and aggregation for all six fusion methods.

- [ ] **Step 3: Update catalogue paths**

Point every Stage 5 joint experiment to the shared model and its specific fusion module while retaining the historical wrapper path used by old runs.

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest tests/models tests/test_repository_documentation.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add docs/model_architecture.md docs/experiment_catalogue.md README.md tests/models/test_model_documentation.py
git commit -m "docs: explain shared multiview model architecture"
```
