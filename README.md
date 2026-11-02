# Anatomy-aware weak localization in chest X-ray vision–language models

Research code for **Improving Weak Localization in Chest X-Ray Vision–Language Models Using Anatomy-Aware Refinement**, a Master's thesis by **Harshvardhan Agarwal**, Friedrich-Alexander-Universität Erlangen-Nürnberg, 2026.

This repository extends ALBEF to study whether anatomical guidance and original/lung/heart representations improve recognition and the spatial evidence used by chest X-ray vision–language models. The principal findings are **Cardiomegaly** and **Pleural Effusion**. A strong image-level classifier does not necessarily localize the finding accurately.

The structural migration places 130 original Python implementations and 45 configurations in the research hierarchy below, without changing model/method logic or historical config values. Original IO-helper snapshots are retained for the two optional path-routing revisions. Historical paths remain available through compatibility loaders and config symlinks. Seven output-free notebooks and 34 aggregate historical results are also organized with legacy aliases. Output-bearing notebooks, per-image predictions, data and resource-coupled upstream tools remain in place. Full thesis reproduction remains dependent on the run artifacts listed in [known discrepancies](docs/discrepancies.md).

## Find the model code

**[Open the model-code index](docs/model_code.md)** for direct links to the full implementation of every main experiment. Start here when reading the research code, rather than the historical root compatibility loaders.

- [A0 baseline](experiments/baseline/README.md)
- [A1–A4 guidance and A3.1/A3.2](experiments/anatomy_guidance/README.md)
- [Original/lung/heart models](experiments/single_view/README.md)
- [FC, mean, gated, residual, Transformer and cross-attention fusion](experiments/joint_fusion/README.md)
- [Late score fusion](experiments/late_fusion/README.md)

A0–A4 share the ALBEF architecture; their guidance differences are explicitly mapped to trainer/loss code. The six fusion architectures remain separate full source files.

## Start here

| Guide | Contents |
|---|---|
| [Methodology](docs/methodology.md) | Anatomical guidance, six fusion models, attribution and evaluation |
| [Experiments](docs/experiments.md) | Thesis stage to source/configuration mapping and historical variants |
| [Reproduction](docs/reproduction.md) | Environments, inputs, launch conventions and validation gates |
| [Local asset/path overrides](docs/local_paths.md) | Optional evaluation assets and new path-bound local config copies |
| [External dependencies](docs/external_dependencies.md) | Upstream code, pretrained assets, licenses and missing pins |
| [Data access](docs/data_access.md) | Dataset roles, manifests and restricted artifacts |
| [Results and provenance](docs/results_provenance.md) | Thesis-reported findings and evidence needed to reproduce them |
| [Known discrepancies](docs/discrepancies.md) | Thesis, current checkout and historical implementation differences |
| [Repository architecture](docs/repository_architecture.md) | Applied layout, compatibility and remaining migration scope |
| [Migration validation](docs/migration_validation.md) | Exact-byte preservation and numerical compatibility checks |
| [Notebook catalog](notebooks/README.md) | Grouped analyses, canonical viewing paths and execution context |
| [Historical results](results/README.md) | Aggregate summaries and held per-image prediction files |
| [Agent instructions](AGENTS.md) | Constraints for future automated changes |

## Experimental pipeline

1. **Establish baselines:** original-image ALBEF with positive–negative ITC-margin Grad-CAM; external BioViL-T grounding and supplied CheXzero comparisons.
2. **Apply direct anatomical guidance:** A0 reference, A1 support guidance, A2 uncertainty weighting, A3 positive-only support and coefficient ablations, and A4 classification preservation. Guidance models are pathology-specific.
3. **Train anatomical single views:** original radiograph, lung-only and heart-only images from the same radiograph.
4. **Combine views:** normalized score ensembles or jointly trained FC, mean, gated, residual, two-layer Transformer and anchored cross-attention models.
5. **Evaluate:** held-out ROC-AUC and validation-thresholded F1; pointing-game peak alignment; matched qualitative figures.

A3.1 and A3.2 use support coefficients 0.005 and 0.001; A3 uses 0.01. A5/A6 are preserved historical work outside the main thesis workflow. Their presence does not make them thesis reproduction requirements. No implemented semi-supervised training stage is claimed.

## Datasets and assets

- **MIMIC-CXR:** paired image–text training records.
- **VinDr-CXR:** a fixed 2,000-image training-data validation subset and the official 3,000-image test set, as described in the thesis.
- **CheXmask:** lung and heart masks used to construct anatomical views.
- **Pretrained assets:** BERT/DeiT/ALBEF assets as required by the particular run; BioViL-T and historical anatomical segmentation have additional model dependencies.

The original checked-in MIMIC manifest has 215,394 records. History describes excluding 43 missing-mask records for the matched anatomical-view population, consistent with the thesis count of 215,351. The generated manifest itself is not included, so exact run membership still needs verification. A0–A2 used full datasets according to the researcher and historical trainer behavior; old subset YAML settings alone do not describe those earlier runs.

Use authorized local data. **Do not publish patient images, report-bearing manifests, private datasets or checkpoints.** Some historical previews, manifests and notebook outputs are already tracked and require a publication review; this documentation does not certify them as redistributable.

## Current repository layout

```text
src/cxr_vlm/          Reusable models, datasets, anatomy, evaluation, optimizers
experiments/         Baseline, guidance, single-view, late/joint fusion and extraction
configs/historical/  Original configurations preserved verbatim
configs/models/      BERT architecture configuration
tools/               Preprocessing, inspection, diagnostics and experiment launcher
archive/             Preserved A5/A6, exploratory localization and upstream tasks
notebooks/           Comparisons, evaluation and figures; held outputs remain at old paths
results/historical/  Aggregate historical outputs, preserved without value changes
results/             Legacy aliases and held per-image predictions
docs/                Thesis guides, provenance and applied migration manifest
Pretrain*.py, etc.    Historical compatibility loaders, not duplicate implementations
```

The generic-looking `Pretrain_A0_multiview_dual_auc.py` trains configurable **single-view** ALBEF. The generic-looking multi-view bare-prompt evaluator currently constructs the **cross-attention** model. See the experiment guide before choosing an entry point.

## Installation and execution

There is no single verified environment covering every historical script. The [reproduction guide](docs/reproduction.md) distinguishes legacy ALBEF, newer fusion, BioViL-T, segmentation and analysis environments.

For a **CPU legacy smoke environment**, Python 3.9.25 with PyTorch 1.8.0+cpu and torchvision 0.9.0+cpu was validated during onboarding. The installation recipe is in the guide. It does not validate GPU training, newer Pillow-dependent transforms or full thesis reproduction.

From the repository root, after activating a compatible environment:

```bash
python -B tools/run_experiment.py experiments/baseline/Pretrain.py --help
python -B tools/run_experiment.py experiments/evaluation/evaluate_zero_shot_f1_from_validation.py --help
```

These commands check CLI imports, not training. Use the registered launcher from the repository root; directly running a deeply nested canonical file does not restore historical sibling imports. Legacy commands are retained. Configuration symlinks require filesystem/Git symlink support (validated on Linux). Dataset/checkpoint execution templates and required prerequisites are documented separately. Do not launch a large run from an unverified historical YAML: augmentation, checkpoint-selection and prepared-data differences remain unresolved for some reported runs.

## Evaluation and reported findings

The primary quantitative localization endpoint is a **pointing-game hit**: the saved map's selected maximum falls inside a target-pathology box. Classification, peak alignment and visual appearance answer different questions. Historical FROC and anatomy-gating explorations are preserved but are not interchangeable with this endpoint.

Selected **thesis-reported**, not independently reproduced, outcomes:

| Outcome | Reported result | Thesis source |
|---|---|---|
| Original A0 Pleural Effusion ROC-AUC | 0.9788 | Table 6.1 |
| Anchored cross-attention Cardiomegaly ROC-AUC / F1 | 0.9427 / 0.6125 | Table 6.7 |
| Validation-tuned late-fusion Pleural Effusion F1 | 0.7333 | Table 6.6 |
| Gated fusion Cardiomegaly pointing-game rate | 82.20% | Table 6.8 |
| Anchored cross-attention Pleural Effusion pointing-game rate | 17.12% | Table 6.8 |

These observations do not establish a universal advantage of anatomical restriction or fusion. Classification differences are descriptive; the paired localization tests have their own defined scope. See [results provenance](docs/results_provenance.md).

## Thesis reference and upstream attribution

Harshvardhan Agarwal. *Improving Weak Localization in Chest X-Ray Vision–Language Models Using Anatomy-Aware Refinement*. Master's thesis in Computer Science, Friedrich-Alexander-Universität Erlangen-Nürnberg, 2026. The report was supplied privately for this audit; no public thesis URL or DOI has been verified, and the PDF is not added here.

The inherited ALBEF implementation is associated with Junnan Li et al., *Align before Fuse: Vision and Language Representation Learning with Momentum Distillation*, NeurIPS 2021 ([paper](https://arxiv.org/abs/2107.07651)). Retain its [Salesforce license](LICENSE.txt), source notices and other third-party attributions. The prior root README is preserved verbatim as [historical upstream documentation](docs/upstream/README.original.txt); its task commands are not the thesis run protocol.
