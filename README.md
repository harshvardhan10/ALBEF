# Anatomy-Aware Weak Localization in Chest X-Ray Vision–Language Models

Research code for **Improving Weak Localization in Chest X-Ray Vision–Language Models Using Anatomy-Aware Refinement**

**Harshvardhan Agarwal · Master's thesis · Friedrich-Alexander-Universität Erlangen-Nürnberg · 2026**

> **`work` is the primary research and development branch.** Use this branch for the organized model implementations, experiment guides and reproduction instructions. The `main` branch preserves the earlier repository state and points here.

## Research motivation

Chest X-ray vision–language models can recognize a finding without grounding their predictions in the relevant anatomy. This thesis investigates whether anatomical guidance and representations of the original image, lungs and heart improve classification and weak localization, with a focus on **Cardiomegaly** and **Pleural Effusion**.

The repository extends ALBEF with anatomy-aware training, anatomical single-view models, late score fusion and six distinct jointly trained fusion architectures. It also contains heatmap extraction, localization evaluation and external baseline comparisons. Classification performance, localization accuracy and qualitative appearance are evaluated separately.

## Find the model code

**Start with the [model-code index](docs/model_code.md)** for direct links to full architectures, training logic and experiment-specific differences.

## Experimental pipeline

1. **Establish baselines:** evaluate original-image ALBEF and external comparisons; extract positive–negative ITC-margin Grad-CAM maps.
2. **Apply anatomical guidance:** compare A0 with support guidance, uncertainty weighting, positive-only support, support-coefficient ablations and classification preservation.
3. **Train anatomical views:** learn from original, lung-only and heart-only representations of the same radiograph.
4. **Combine views:** compare late score fusion with jointly trained FC, mean, gated, residual, Transformer and anchored cross-attention models.
5. **Evaluate classification and localization:** assess ROC-AUC, validation-thresholded F1, pointing-game peak alignment and matched qualitative examples.

Guidance experiments are pathology-specific. A3 uses a support coefficient of 0.01; A3.1 and A3.2 use 0.005 and 0.001.

## Datasets and pretrained assets

| Dataset or asset | Role |
|---|---|
| MIMIC-CXR | Paired image–text training |
| VinDr-CXR | Classification and localization evaluation; the thesis describes a fixed 2,000-image validation subset from the training set and the official 3,000-image test set |
| CheXmask | Lung and heart masks for anatomical guidance and view construction |
| BERT, DeiT and ALBEF assets | Initialization and tokenization, depending on the experiment |
| External model assets | Additional requirements for BioViL-T and historical anatomical segmentation |

Obtain datasets through their authorized access procedures and keep local data and checkpoints outside the public repository. Dataset access, manifest roles and cohort uncertainties are documented in [data access](docs/data_access.md).

## Repository layout

```text
src/cxr_vlm/          Reusable models, datasets, anatomy, evaluation and optimization
experiments/         Baseline, guidance, single-view, fusion and evaluation entry points
configs/historical/  Original experiment configurations
configs/models/      BERT architecture configuration
tools/               Experiment launcher, preprocessing, inspection and diagnostics
compatibility/       Grouped historical root command wrappers
notebooks/           Analysis, comparisons, evaluation and figures
results/             Historical summaries, compatibility aliases and existing outputs
archive/             Non-thesis experiments, upstream tasks and maintenance history
docs/                Methodology, experiment mapping, reproduction and provenance
```

The full implementations live in `src/cxr_vlm/` and the experiment directories.

## Installation and quick start

Clone the primary research branch:

```bash
git clone --branch work --single-branch https://github.com/harshvardhan10/ALBEF.git
cd ALBEF
```

After activating a compatible environment, run from the repository root:

```bash
# Inspect the baseline training interface.
python -B tools/run_experiment.py experiments/baseline/Pretrain.py --help

# Inspect classification evaluation.
python -B tools/run_experiment.py experiments/evaluation/evaluate_zero_shot_f1_from_validation.py --help

# Run the CPU compatibility checks without private data or checkpoints.
python -B -m unittest discover -s tests/compatibility -v
```

The launcher accepts canonical paths and historical entrypoint names. For example, `python tools/run_experiment.py Pretrain.py --help` selects the baseline trainer. Former commands such as `python Pretrain.py` require the launcher prefix. 

## Evaluation and thesis-reported results

The primary localization endpoint is a **pointing-game hit**: the selected maximum of a saved heatmap falls inside a target-pathology bounding box.

The following values are **reported in the thesis**:

| Experiment and endpoint | Reported result | Thesis source |
|---|---|---|
| Original A0: Pleural Effusion ROC-AUC | 0.9788 | Table 6.1 |
| Anchored cross-attention: Cardiomegaly ROC-AUC / F1 | 0.9427 / 0.6125 | Table 6.7 |
| Validation-tuned late fusion: Pleural Effusion F1 | 0.7333 | Table 6.6 |
| Gated fusion: Cardiomegaly pointing-game rate | 82.20% | Table 6.8 |
| Anchored cross-attention: Pleural Effusion pointing-game rate | 17.12% | Table 6.8 |

The results do not establish a universal advantage of anatomical restriction or fusion. Classification differences are descriptive, and the paired localization tests have a defined comparison scope.

## Documentation

- [Methodology](docs/methodology.md): anatomical guidance, fusion, attribution and evaluation.
- [Experiments](docs/experiments.md): thesis stages mapped to implementations and configurations.
