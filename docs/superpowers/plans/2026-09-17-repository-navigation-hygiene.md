# Repository Navigation and Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ALBEF repository immediately understandable and safe for supervisor review without changing experiment behavior.

**Architecture:** Keep ALBEF as the primary model repository, link the separate SLURM and CheXzero repositories, and organize documentation around the six confirmed experimental stages. Remove generated clutter and large derived inputs while preserving scientific provenance and upstream attribution.

**Tech Stack:** Markdown, Python 3.8.20, Git, Jupyter nbformat, PyYAML

**Spec:** `docs/superpowers/specs/2026-09-17-thesis-repository-cleanup-design.md`

## Global Constraints

- Preserve the six-stage thesis narrative exactly as approved.
- Preserve upstream Salesforce ALBEF attribution and `LICENSE.txt`.
- Do not claim quantitative localization results.
- Record cross-attention fusion as ongoing.
- Record non-multiview dual-checkpoint evaluation as pending.
- Do not rewrite Git history.
- Do not modify `main` directly.
- Keep CheXzero external at commit `5f3997d8456fc2fe75742018557d79793cc546aa`.
- Keep cluster SLURM scripts in `harshvardhan10/thesis-codebase`.
- Support Python 3.8.20, PyTorch 1.8.1+cu111, and CUDA 11.1.

---

### Task 1: Add repository-documentation validation

**Files:**
- Create: `tests/test_repository_documentation.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Consumes: repository paths relative to the repository root.
- Produces: fast tests that enforce the six stages, companion links, localization caveat, and required documentation files.

- [ ] **Step 1: Write the failing documentation test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_repository_documents_exist():
    required = [
        "README.md",
        "docs/experiment_catalogue.md",
        "docs/multiview_findings.md",
        "docs/reproduction.md",
        "docs/checkpoints.md",
        "docs/limitations.md",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    assert not missing, f"Missing documentation: {missing}"


def test_readme_contains_authoritative_thesis_navigation():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for stage in range(1, 7):
        assert f"Stage {stage}" in text
    assert "harshvardhan10/CheXzero" in text
    assert "5f3997d8456fc2fe75742018557d79793cc546aa" in text
    assert "harshvardhan10/thesis-codebase" in text
    assert "quantitative localization" in text.lower()
    assert "ongoing" in text.lower()
```

- [ ] **Step 2: Run the test and confirm the new documentation requirements fail**

Run:

```bash
python -m pytest tests/test_repository_documentation.py -v
```

Expected: failure because the thesis documentation files do not yet exist and the README is still the upstream landing page.

- [ ] **Step 3: Commit the failing contract test**

```bash
git add tests/__init__.py tests/test_repository_documentation.py
git commit -m "test: define thesis repository documentation contract"
```

### Task 2: Replace the landing page and add the experiment catalogue

**Files:**
- Modify: `README.md`
- Create: `docs/experiment_catalogue.md`
- Create: `docs/checkpoints.md`

**Interfaces:**
- Consumes: the approved six stages and the current repository inventory.
- Produces: the primary supervisor navigation and experiment-to-code mapping.

- [ ] **Step 1: Rewrite `README.md` with this fixed section order**

```markdown
# Improving Weak Localization in Chest X-Ray Vision-Language Models

## Research question
## Current conclusion
## Experimental stages
### Stage 1: Establish localization baselines
### Stage 2: Investigate direct anatomical guidance
### Stage 3: Investigate learned spatial refinement
### Stage 4: Establish anatomy-focused single-view baselines
### Stage 5: Investigate combinations of anatomical views
### Stage 6: Evaluate and interpret the findings
## Repository map
## Reproduction
## Environment
## Results and limitations
## Upstream ALBEF attribution
```

The current conclusion must say that classification remains strong but multiview fusion did not consistently improve weak localization in qualitative comparisons. It must not report a quantitative localization ranking.

- [ ] **Step 2: Create one catalogue row per canonical experiment**

Use this exact schema:

```markdown
| ID | Stage | Question | Canonical trainer | Model | Configuration | Checkpoint roles | Result status | Experiment status |
|---|---:|---|---|---|---|---|---|---|
```

Include A0-A4, A5, A6.1, original, lung-only, heart-only, score ensemble, heatmap ensemble, learned ensemble, FC, mean, gated, residual, Transformer, and cross-attention. Cross-attention status is `ongoing`. Non-multiview dual-checkpoint evaluation status is `pending`.

- [ ] **Step 3: Document checkpoint semantics**

`docs/checkpoints.md` must define:

```text
checkpoint_best_cardiomegaly_auc.pth
checkpoint_best_macro_auc_stable.pth
checkpoint_last.pth
```

State that the first two are separate scientific checkpoint roles and that `checkpoint_last.pth` is for crash recovery. List which completed multiview results already exist and which experiments still require both evaluations.

- [ ] **Step 4: Run the documentation tests**

Run:

```bash
python -m pytest tests/test_repository_documentation.py -v
```

Expected: remaining failures only for documents created in later tasks.

- [ ] **Step 5: Commit the supervisor navigation**

```bash
git add README.md docs/experiment_catalogue.md docs/checkpoints.md
git commit -m "docs: add thesis navigation and experiment catalogue"
```

### Task 3: Document findings, reproduction, and limitations

**Files:**
- Create: `docs/multiview_findings.md`
- Create: `docs/reproduction.md`
- Create: `docs/limitations.md`

**Interfaces:**
- Consumes: experiment catalogue, pinned companion repositories, confirmed environment.
- Produces: an honest findings narrative and reproducible repository boundaries.

- [ ] **Step 1: Write the supported multiview findings**

Include these claims with qualitative scope:

```markdown
- Multiview fusion did not consistently improve weak localization.
- FC and mean fusion occasionally produced sharper or better-aligned pleural-effusion activation.
- Gated, residual, and Transformer fusion frequently produced central, diffuse, or off-target activation.
- The original single-view ALBEF remained qualitatively competitive.
- More complex representation fusion did not force spatially accurate evidence because the training objectives did not directly supervise lesion localization.
- Cross-attention results are excluded while the experiment is ongoing.
```

Do not add numbers until the user's local result artifacts are supplied and verified.

- [ ] **Step 2: Write reproduction boundaries**

`docs/reproduction.md` must contain:

```markdown
## Model and evaluation code
This repository: https://github.com/harshvardhan10/ALBEF

## Cluster launch scripts
https://github.com/harshvardhan10/thesis-codebase

## CheXzero baseline
https://github.com/harshvardhan10/CheXzero
Pinned commit: 5f3997d8456fc2fe75742018557d79793cc546aa

## Supported environment
Python 3.8.20
PyTorch 1.8.1+cu111
CUDA 11.1
```

Add command templates that use explicit config, output directory, and checkpoint arguments. Do not include personal cluster paths.

- [ ] **Step 3: Write limitations**

Explicitly record:

- no final quantitative localization evaluation;
- heatmaps were visually poor and conclusions remain qualitative;
- full training cannot be reproduced without controlled MIMIC-CXR access;
- configuration comments were historically used to switch full and subset runs;
- cross-attention remains ongoing;
- bone suppression remains future work;
- the semi-supervised title component is outside the current repository claims.

- [ ] **Step 4: Run the complete documentation test**

Run:

```bash
python -m pytest tests/test_repository_documentation.py -v
```

Expected: all documentation tests pass.

- [ ] **Step 5: Commit findings and reproduction documentation**

```bash
git add docs/multiview_findings.md docs/reproduction.md docs/limitations.md
git commit -m "docs: document findings reproduction and limitations"
```

### Task 4: Remove generated Python artifacts and add ignore rules

**Files:**
- Create: `.gitignore`
- Delete: every tracked `*.pyc` file and every tracked file below `__pycache__/`
- Test: `tests/test_repository_hygiene.py`

**Interfaces:**
- Consumes: tracked-file list.
- Produces: repository hygiene rules and a regression test preventing generated caches from returning.

- [ ] **Step 1: Write the failing hygiene test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_python_cache_artifacts_are_tracked_in_tree():
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and (path.suffix == ".pyc" or "__pycache__" in path.parts)
    ]
    assert offenders == []


def test_large_mimic_manifest_is_not_present():
    assert not (ROOT / "data" / "mimic_cxr.json").exists()
    assert (ROOT / "data" / "mimic_cxr.example.json").is_file()
```

- [ ] **Step 2: Run the hygiene test and confirm failure**

Run:

```bash
python -m pytest tests/test_repository_hygiene.py -v
```

Expected: failure listing tracked bytecode and the current MIMIC manifest.

- [ ] **Step 3: Add ignore rules**

```gitignore
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.pytest_cache/
.coverage
htmlcov/
output*/
checkpoints/
*.pth
*.pt
*.ckpt
data/mimic_cxr.json
results/raw/
previews/
```

- [ ] **Step 4: Remove only tracked generated cache artifacts**

Run:

```bash
git ls-files -z '*.pyc' '*__pycache__*' | xargs -0 -r git rm
```

Review the staged list and confirm every removed file is generated bytecode.

- [ ] **Step 5: Commit cache cleanup**

```bash
git add .gitignore tests/test_repository_hygiene.py
git commit -m "chore: remove generated Python cache artifacts"
```

### Task 5: Replace the tracked MIMIC manifest with a safe example

**Files:**
- Delete: `data/mimic_cxr.json`
- Create: `data/mimic_cxr.example.json`
- Create: `docs/data_preparation.md`
- Modify: `tests/test_repository_documentation.py`

**Interfaces:**
- Consumes: the manifest schema used by training datasets.
- Produces: a small non-sensitive example and instructions pointing to the preparation script in `thesis-codebase`.

- [ ] **Step 1: Create a schema-only example**

```json
[
  {
    "image": "/path/to/authorized/mimic-cxr/image.jpg",
    "caption": "Example report text for schema illustration only."
  }
]
```

- [ ] **Step 2: Write data preparation instructions**

Document that users must obtain MIMIC-CXR through its controlled-access process and run:

```text
thesis-codebase/scripts/prepare_mimic_albef.py
thesis-codebase/scripts/prepare_mimic_albef.slurm
```

State that real manifests, reports, image paths, and derived patient-level data must remain outside Git.

- [ ] **Step 3: Remove the large manifest from the current tree**

Run:

```bash
git rm data/mimic_cxr.json
```

Do not purge prior Git history in this task.

- [ ] **Step 4: Add `docs/data_preparation.md` to the required-document test and run it**

Run:

```bash
python -m pytest tests/test_repository_documentation.py tests/test_repository_hygiene.py -v
```

Expected: pass.

- [ ] **Step 5: Commit data hygiene**

```bash
git add data/mimic_cxr.example.json docs/data_preparation.md tests/test_repository_documentation.py
git commit -m "docs: replace MIMIC manifest with preparation guidance"
```

### Task 6: Separate runtime and development dependencies

**Files:**
- Modify: `requirements_albef_legacy.txt`
- Create: `requirements-eval.txt`
- Create: `requirements-dev.txt`
- Modify: `requirements.txt`
- Modify: `docs/reproduction.md`
- Test: `tests/test_environment_files.py`

**Interfaces:**
- Consumes: confirmed legacy training environment.
- Produces: explicit training, evaluation, and development dependency entry points.

- [ ] **Step 1: Write dependency-file tests**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_environment_files_are_present_and_document_legacy_runtime():
    legacy = (ROOT / "requirements_albef_legacy.txt").read_text()
    assert "torch==1.8.1+cu111" in legacy
    assert "torchvision==0.9.1+cu111" in legacy
    assert (ROOT / "requirements-eval.txt").is_file()
    assert (ROOT / "requirements-dev.txt").is_file()


def test_development_requirements_include_test_runner():
    assert "pytest" in (ROOT / "requirements-dev.txt").read_text()
```

- [ ] **Step 2: Run the test and confirm failure**

Run:

```bash
python -m pytest tests/test_environment_files.py -v
```

Expected: failure because the confirmed runtime and split dependency files are not yet represented.

- [ ] **Step 3: Normalize dependency roles**

Use `requirements_albef_legacy.txt` for the pinned training stack, `requirements-eval.txt` for NumPy, pandas, scikit-learn, Pillow, OpenCV, matplotlib, and PyYAML, and `requirements-dev.txt` for pytest and notebook tooling.

Make `requirements.txt` a documented convenience file that references the three roles with `-r` lines. Preserve package versions required by successful training.

- [ ] **Step 4: Run dependency tests**

Run:

```bash
python -m pytest tests/test_environment_files.py -v
```

Expected: pass.

- [ ] **Step 5: Commit dependency separation**

```bash
git add requirements.txt requirements_albef_legacy.txt requirements-eval.txt requirements-dev.txt docs/reproduction.md tests/test_environment_files.py
git commit -m "build: document legacy training and evaluation environments"
```

### Task 7: Curate notebooks and record legacy files

**Files:**
- Create: `docs/notebook_index.md`
- Create: `docs/legacy_inventory.md`
- Modify: tracked notebooks selected in `docs/notebook_index.md`
- Move: scientifically relevant superseded scripts into `legacy/` only after their canonical replacement exists

**Interfaces:**
- Consumes: notebook sizes, experiment catalogue, canonical-file mapping.
- Produces: a small curated notebook set with outputs removed and an explicit legacy inventory.

- [ ] **Step 1: Generate a notebook size and output report**

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

for path in sorted(Path("notebooks").glob("*.ipynb")):
    data = json.loads(path.read_text(encoding="utf-8"))
    outputs = sum(len(cell.get("outputs", [])) for cell in data.get("cells", []))
    print(f"{path}	{path.stat().st_size}	outputs={outputs}")
PY
```

- [ ] **Step 2: Classify notebooks**

`docs/notebook_index.md` must label each notebook as:

- curated result presentation;
- reproducible evaluation;
- exploratory;
- superseded.

Retain the matched ALBEF/BioViL-T comparison, multiview comparison, classification comparison, and loss-curve notebooks as curated entries.

- [ ] **Step 3: Strip outputs from curated reproducible notebooks**

Run for each selected notebook:

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/NOTEBOOK.ipynb
```

Replace `NOTEBOOK.ipynb` in each command with a path explicitly listed in `docs/notebook_index.md`. Do not strip an output that is the only surviving scientific result until the result is exported into `results/summaries/`.

- [ ] **Step 4: Record legacy status before moving files**

`docs/legacy_inventory.md` must list old path, replacement path, reason for supersession, and whether a compatibility wrapper remains.

- [ ] **Step 5: Re-run repository tests**

Run:

```bash
python -m pytest tests/test_repository_documentation.py tests/test_repository_hygiene.py tests/test_environment_files.py -v
```

Expected: pass.

- [ ] **Step 6: Commit notebook and legacy curation**

```bash
git add notebooks docs/notebook_index.md docs/legacy_inventory.md results/summaries
git commit -m "chore: curate notebooks and document legacy artifacts"
```
