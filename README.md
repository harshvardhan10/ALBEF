# Anatomy-aware chest X-ray vision-language experiments

Thesis code for **Improving Weak Localization in Chest X-Ray Vision-Language Models Using Anatomy-Aware and Semi-Supervised Refinement**. This repository asks whether anatomical views, guidance, patch refinement and multiview fusion improve weak localization while preserving zero-shot classification.

This cleanup makes the implementation reviewable for the thesis supervisor. The title is historical: no new semi-supervised method or final quantitative localization ranking is claimed here.

## Start here

- [Supervisor reading guide](docs/supervisor_review_guide.md)
- [Experiment catalogue](docs/experiment_catalogue.md): hypotheses, code, configurations and launch records
- [Model architecture](docs/model_architecture.md): differences among the six fusion variants
- [Cross-attention validation](docs/cross_attention_validation.md): precise server commands for the completed training run
- [Reproduction](docs/reproduction.md), [checkpoints](docs/checkpoints.md) and [limitations](docs/limitations.md)
- [Source provenance](docs/source_provenance.md) and [cleanup progress](docs/superpowers/cleanup-progress.md)

## Six experimental stages

1. Establish ALBEF, BioViL-T and CheXzero classification/localization baselines.
2. Investigate direct anatomy guidance (A0–A4).
3. Investigate learned spatial refinement (A5/A6.1).
4. Compare original, lung-only and heart-only views.
5. Compare prediction ensembles, heatmap ensembles and joint view fusion.
6. Interpret quantitative classification and matched qualitative localization, including failures.

Cross-attention training is **complete, evaluation pending**, as confirmed by the author on 2026-10-20. Checkpoint files and results remain on the server. The original model and trainer implementations remain intact while baseline validation is pending.

## Repository map

| Location | Responsibility |
|---|---|
| models/, dataset/, root Pretrain*.py | Existing ALBEF research implementations and training entry points |
| configs/, configs_new/ | Historical methodology/configuration evidence; verify resolved run config |
| scripts/, scripts_new/ | Evaluation and heatmap methods |
| cluster/historical/ | Pinned cluster launch records from thesis-codebase; site-specific, not portable |
| cluster/data_preparation/ | Imported Python preparation utilities |
| cluster/slurm/ | New explicit-input validation launcher |
| baselines/chexzero/ | Pinned CheXzero code, vocabulary, LICENSE and NOTICE |
| notebooks/, notebooks/data_preparation/ | Existing analysis and imported preparation notebooks |
| docs/, tests/ | Review narrative, provenance and focused CPU checks |

## Environment and evidence

The documented server training stack is Python 3.8.20, PyTorch 1.8.1+cu111 and CUDA 11.1. Do not replace it with the notebook dependency freeze. CPU tests run locally do not establish GPU, checkpoint or distributed-training equivalence. See [validation status](docs/cleanup_validation.md).

## Upstream attribution

Built on [Salesforce ALBEF](https://github.com/salesforce/ALBEF), *Align before Fuse: Vision and Language Representation Learning with Momentum Distillation*. Preserve the root LICENSE. The [original ALBEF README](docs/upstream-albef-readme.md) is retained. CheXzero's separate license and attribution are retained under baselines/chexzero.
