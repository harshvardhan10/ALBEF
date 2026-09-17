# Thesis Repository Cleanup Design

**Date:** 2026-09-17  
**Repository:** `harshvardhan10/ALBEF`  
**Branch:** `thesis-repository-cleanup`

## 1. Purpose

This cleanup will make the repository reviewable as the primary model-code repository for the thesis:

> Improving Weak Localization in Chest X-Ray Vision-Language Models Using Anatomy-Aware and Semi-Supervised Refinement

The implementation and documentation will focus on the actual completed work. The semi-supervised component will not be interpreted or claimed in this cleanup.

The central thesis question is whether anatomy-aware input representations, anatomical guidance, learned spatial refinement, and multiview fusion improve weak localization while preserving zero-shot classification.

## 2. Success criteria

The cleanup is successful when a supervisor can:

1. understand the six experimental stages from the repository landing page;
2. locate the canonical model, trainer, configuration, and evaluation entry point for each experiment;
3. distinguish completed, failed, pending, and ongoing experiments;
4. inspect each fusion architecture without reading duplicated ALBEF implementations;
5. reproduce the cluster command through the companion SLURM repository;
6. identify the checkpoint and evaluation settings behind each reported result;
7. understand that localization conclusions are qualitative and that no final quantitative localization ranking is claimed.

## 3. Confirmed experimental narrative

### Stage 1: Establish localization baselines

**Question:** How well do existing models recognize and localize pathology?

This stage contains:

- ALBEF zero-shot classification and explicitly identified ALBEF heatmap methods;
- BioViL-T native phrase grounding;
- CheXzero pathology-specific localization from `harshvardhan10/CheXzero`, pinned to commit `5f3997d8456fc2fe75742018557d79793cc546aa`.

CheXzero remains in its own repository. The ALBEF repository records the pinned external implementation and comparable results without duplicating its code.

### Stage 2: Investigate direct anatomical guidance

**Question:** Can anatomical constraints guide the model toward relevant regions?

This stage contains:

- A0: control without anatomy regularization;
- A1: anatomy support guidance;
- A2: uncertainty-weighted support guidance;
- A3: positive-only support guidance;
- A4: positive-only support guidance with classification preservation.

A4 is the final Stage 2 experiment because it tests whether the anatomy objective can be added without losing the original model's classification behavior.

### Stage 3: Investigate learned spatial refinement

**Question:** Can a learned patch head improve the localization signal?

This stage contains:

- A5: identity-initialized patch head trained from detached ALBEF attention while ALBEF continues standard training;
- A6.1: frozen A0 ALBEF with only the patch head trained;
- patch-head heatmap extraction and failure analysis.

The documentation will distinguish pathology-specific refinement from reproducing a generic anatomical support prior.

### Stage 4: Establish anatomy-focused single-view baselines

**Question:** What happens to recognition and grounding when anatomical emphasis changes?

This stage compares:

- original CXR;
- lung-only CXR;
- heart-only CXR.

All views use matched protocols and CheXmask-derived anatomical masks. Classification results include cardiomegaly and pleural effusion.

### Stage 5: Investigate combinations of anatomical views

**Question:** Does combining views improve grounding while preserving recognition?

This stage separates:

1. prediction-score ensembles;
2. heatmap ensembles;
3. validation-fitted learned ensembles;
4. joint multiview pretraining.

Joint fusion variants are:

- fully connected fusion;
- mean fusion;
- gated fusion;
- residual fusion;
- Transformer fusion;
- cross-attention fusion.

Cross-attention is currently running. It remains documented as ongoing until results are available. Its current implementation must not be silently changed or represented as a completed result.

### Stage 6: Evaluate and interpret the findings

**Question:** Which approaches improve localization, under what conditions, and at what classification cost?

This stage combines:

- quantitative classification results using AUC and validation-selected F1;
- matched qualitative heatmap comparisons;
- ablations and method comparisons;
- failure analysis;
- classification-localization trade-offs;
- limitations.

Final quantitative localization evaluation has not been completed because the ALBEF heatmaps are too poor to support a meaningful comparison. FROC utilities remain exploratory and do not support final thesis conclusions.

## 4. Repository boundaries

### Primary repository: ALBEF

The ALBEF repository owns:

- thesis model implementations;
- dataset and transformation code;
- experiment configurations;
- training workflows;
- classification and localization evaluation;
- experiment documentation;
- curated result summaries.

### Cluster repository: thesis-codebase

`harshvardhan10/thesis-codebase` remains the source of cluster-specific SLURM scripts and data-preparation jobs.

The repositories will cross-link by experiment identifier. SLURM scripts will not be copied into ALBEF during the first cleanup pass.

### External baseline repository: CheXzero

`harshvardhan10/CheXzero` remains independent and is referenced at the pinned thesis commit. Its implementation is not vendored into ALBEF.

## 5. Target repository structure

```text
ALBEF/
├── README.md
├── docs/
│   ├── experiment_catalogue.md
│   ├── multiview_findings.md
│   ├── reproduction.md
│   ├── checkpoints.md
│   ├── limitations.md
│   └── superpowers/specs/
├── src/cxr_albef/
│   ├── models/
│   │   ├── albef.py
│   │   ├── multiview_albef.py
│   │   ├── patch_refinement.py
│   │   └── fusion/
│   │       ├── fc.py
│   │       ├── mean.py
│   │       ├── gated.py
│   │       ├── residual.py
│   │       ├── transformer.py
│   │       └── cross_attention.py
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── localization/
├── configs/
│   ├── experiments/
│   │   ├── stage_02_anatomy_guidance/
│   │   ├── stage_03_patch_refinement/
│   │   ├── stage_04_single_view/
│   │   └── stage_05_multiview/
│   └── environments/
├── scripts/
├── notebooks/
├── results/
│   └── summaries/
├── tests/
└── legacy/
```

This is a target structure, not permission for a blind move. Files will move only when imports, wrappers, and verification protect existing behavior.

## 6. Model consolidation

### Shared ALBEF components

The original encoder, projection, momentum, queue, MLM, ITA, and ITM behavior will be shared. Upstream Salesforce attribution and licensing will remain visible.

### Fusion interface

Each fusion module will accept aligned original, lung, and heart token sequences and return the fused token sequence expected by the shared multiview ALBEF implementation.

The following behavior must remain distinct:

- **FC:** concatenate corresponding view tokens, project, then normalize;
- **Mean:** average corresponding view tokens without learned fusion parameters;
- **Gated:** derive sample-specific view weights from view representations and apply them across tokens;
- **Residual:** preserve the original-view representation while adding gated anatomical-view contributions;
- **Transformer:** process the three representations at each corresponding token position using the existing Transformer fusion design;
- **Cross-attention:** preserve the currently running cross-attention design and its initialization rules.

Each fusion method retains its momentum counterpart and method-specific initialization where the historical implementation requires one.

### Patch refinement

Patch-head refinement remains a separate workflow. It will not be folded into the multiview model or described as a fusion method.

## 7. Training consolidation

The cleanup will create four visible workflows:

1. anatomy guidance for A0-A4;
2. patch refinement for A5 and A6.1;
3. single-view pretraining parameterized by original, lung, or heart view;
4. multiview pretraining parameterized by fusion type.

Existing entry-point filenames will initially remain as compatibility wrappers where the SLURM repository depends on them.

The first consolidation priority is multiview model code because model review is the supervisor's primary need.

## 8. Configuration policy

Experiment configurations become immutable records. A run must not depend on manually commenting or uncommenting a field.

Full-dataset training will be explicit:

```yaml
train_subset_size: null
```

Debug or subset experiments will use separate configurations or explicit command-line overrides.

All completed thesis experiments are treated as full-dataset experiments. Historical configuration ambiguity will be documented when the exact resolved file was not saved.

Machine-specific paths will move out of methodological configurations. Environment files or documented command-line values will supply cluster paths.

## 9. Checkpoint and result provenance

The catalogue records, where applicable:

- best cardiomegaly AUC checkpoint;
- best stable-label macro AUC checkpoint;
- last checkpoint for crash recovery only;
- experiment commit;
- resolved configuration;
- split identity;
- prompt definitions;
- label order;
- classification threshold source;
- evaluation script;
- result artifact.

Multiview experiments have results for the best cardiomegaly AUC and best stable-label macro AUC checkpoints.

Other experiments still require evaluation with both checkpoint roles. They will be marked as pending rather than assigned invented or partial results.

User-provided local result files will be curated into summaries only after they are supplied. Large checkpoints and raw generated outputs will not be committed.

## 10. Localization method provenance

Every localization result records:

- model and experiment;
- method, such as ITC Grad-CAM, ITC-margin Grad-CAM, cross-attention Grad-CAM, patch head, or native phrase grounding;
- score target;
- attention source;
- layer and indexing convention;
- view or branch;
- head aggregation;
- ReLU placement;
- normalization;
- interpolation;
- case-selection rule.

This prevents visually similar heatmaps produced by different objectives from being treated as the same method.

## 11. Documentation deliverables

### README

The README will:

- state the thesis question;
- summarize the six stages;
- show repository boundaries;
- link to the experiment catalogue and reproduction guide;
- report current status without claiming quantitative localization improvement;
- preserve upstream ALBEF attribution.

### Experiment catalogue

The catalogue will map each experiment to its hypothesis, canonical implementation, configuration, checkpoint roles, result availability, status, and supported takeaway.

Statuses are limited to:

- completed;
- completed, evaluation pending;
- ongoing;
- exploratory;
- unsuccessful;
- planned future work.

### Multiview findings

This document will distinguish classification from localization and summarize the supported qualitative conclusion:

- fusion does not consistently improve weak localization;
- some FC and mean cases appear better aligned for pleural effusion;
- gated, residual, and Transformer fusion can produce diffuse, central, or off-target responses;
- the original single-view ALBEF remains qualitatively competitive;
- more complex fusion does not guarantee improved spatial grounding.

Cross-attention findings will be added only after the ongoing experiment completes.

## 12. Repository hygiene

The cleanup will:

- add a thesis-appropriate `.gitignore`;
- remove tracked `.pyc` files and `__pycache__` directories;
- remove exact duplicate utilities and configurations;
- strip large notebook outputs while preserving curated notebooks;
- remove reproducible previews and temporary files;
- stop tracking `data/mimic_cxr.json`;
- provide an example manifest and preparation instructions;
- preserve scientifically relevant superseded code under `legacy/`;
- retain upstream code required by imports;
- avoid Git-history rewriting.

## 13. Environment

The documented successful training environment is:

- Python 3.8.20;
- PyTorch 1.8.1+cu111;
- CUDA 11.1.

Runtime, evaluation, and notebook dependencies will be separated where practical. Dependency cleanup must preserve compatibility with the successful legacy training environment.

## 14. Verification strategy

Full MIMIC-CXR pretraining is outside local verification. The cleanup will use:

- syntax and import checks;
- configuration validation;
- synthetic dataset and transform checks;
- fusion input/output shape checks;
- deterministic old-versus-new fusion comparisons where possible;
- loss and gradient-flow checks;
- momentum-module checks;
- checkpoint missing and unexpected key validation;
- compatibility-wrapper checks;
- cluster smoke-test commands.

The implementation will not claim behavioral equivalence without direct evidence.

## 15. Delivery sequence

1. Create thesis documentation and experiment catalogue.
2. Perform safe repository hygiene.
3. Consolidate multiview fusion modules and shared multiview ALBEF.
4. Consolidate common training and immutable configurations.
5. Organize evaluation and localization code.
6. Add verification tests and cluster smoke-test instructions.
7. Review the branch before any merge into `main`.

## 16. Non-goals

This cleanup will not:

- implement bone suppression;
- invent a semi-supervised method;
- claim quantitative localization results;
- rerun full training locally;
- copy the CheXzero implementation into ALBEF;
- merge cluster-specific SLURM scripts into ALBEF during the first pass;
- rewrite Git history;
- modify `main` directly;
- treat the ongoing cross-attention experiment as complete.
