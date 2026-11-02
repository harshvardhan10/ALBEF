# Methodology

This guide summarizes thesis Chapter 5 and maps its methods to the current source layout. It is a methodology reference, not proof that the latest checked-in defaults reproduce every reported run. Read [discrepancies](discrepancies.md) and [experiments](experiments.md) together.

## Shared model and task

ALBEF uses a local ViT image encoder, a BERT-based text encoder and image-conditioned XBERT fusion. Its baseline objective is the sum of image–text contrastive alignment (ITC, implemented as ITA), image–text matching and masked language modeling. Momentum components and queues are part of the implementations.

- [Single-view model](../src/cxr_vlm/models/model_pretrain.py)
- [ViT](../src/cxr_vlm/models/vit.py)
- [XBERT](../src/cxr_vlm/models/xbert.py)
- [Tokenizer](../src/cxr_vlm/models/tokenization_bert.py)

The main target findings are Cardiomegaly and Pleural Effusion. Anatomy-focused views are masked derivatives of the same radiograph, not separate acquisitions.

## Direct anatomical guidance: A0–A4

The normalized pathology-token attention over image patches is penalized for mass outside a coarse support. Per-sample support weights multiply outside mass; the weighted batch sum is divided by `max(sum(weights), 1)`. Missing target tokens deactivate the support signal.

| Variant | Sample weighting / additional constraint |
|---|---|
| A0 | No anatomical-support penalty |
| A1 | Target phrase occurs in the caption |
| A2 | Exact target caption: 1; target plus uncertainty marker: 0.5; otherwise 0 |
| A3 | Exact target caption only; support coefficient 0.01 |
| A3.1 | A3 with coefficient 0.005 |
| A3.2 | A3 with coefficient 0.001 |
| A4 | Positive-only support plus preservation of frozen-A0 image–text similarity distributions |

Exact-positive means equality after lowercasing/trimming in the implementation. It must not be generalized to arbitrary positive sentences without a separate methodological change. The compatibility name `all_cardiomegaly_captions` maps to the phrase-containing mode.

The thesis uses rectangular supports: Cardiomegaly at 35–75% height and 25–75% width; PE at 55–93% height with bilateral 7–45% and 55–93% width regions. They constrain anatomical plausibility, not lesion shape. A later segmentation-derived heart-prior experiment is distinct from thesis A3.

Sources: [guidance trainer](../experiments/anatomy_guidance/Pretrain_anatomy_prior.py), [A4 trainer](../experiments/anatomy_guidance/Pretrain_anatomy_prior_A4.py), [support loss](../src/cxr_vlm/anatomy/losses.py), [attention extraction](../src/cxr_vlm/anatomy/attention_extract.py), [token selection](../src/cxr_vlm/anatomy/token_utils.py).

Guidance models are trained separately for each target pathology. A4 preserves distributions of global similarities; it does not guarantee identical disease predictions or localization maps. A5/A6 patch-head experiments are historical work outside this thesis scope.

## Original, lung and heart inputs

Original images retain thoracic context. Lung/heart inputs zero pixels outside their corresponding mask, with the mask applied before the base image transform. The thesis specifies RGB, 256×256, ALBEF normalization, conservative affine/photometric augmentation and no random crop or horizontal flip. Newer single-view/fusion paths implement controlled CXR transforms; the checked-in anatomy trainer's generic transform differs, so the exact final anatomy training revision remains unresolved.

CheXmask preparation paths and matched-cohort construction are described in [data access](data_access.md). Do not substitute online decoding, cached masks and precomputed images as equivalent without checking their outputs.

## Score-level combination

Late fusion combines outputs from three separately trained models. Per-view, per-label z-score statistics are fitted on validation predictions. Equal weights are `(1/3, 1/3, 1/3)`; the thesis reports tuned original/lung/heart weights `(0.50, 0.36, 0.14)`. These reported weights are not independently refitted in this documentation phase. Ensemble F1 thresholds are fitted separately on validation data and frozen for test.

Sources: [normalized ensemble evaluator](../experiments/late_fusion/ensemble_zero_shot_scores.py), [validation-fitted weights](../experiments/late_fusion/learn_dual_checkpoint_ensemble_weights.py).

## Joint token fusion

At 256×256 with ViT-B/16, each branch yields 257 tokens of width 768. Fusion operates across corresponding positions; its output is used by the ALBEF objectives. Each model preserves its own momentum pathway.

| Family | Operation | Model |
|---|---|---|
| FC | Concatenation, learned projection, LayerNorm | [FC](../src/cxr_vlm/models/model_pretrain_multiview_fusion.py) |
| Mean | Arithmetic average, no fusion parameters | [Mean](../src/cxr_vlm/models/model_pretrain_multiview_mean_fusion.py) |
| Gated | CLS-conditioned softmax study weights applied across tokens | [Gated](../src/cxr_vlm/models/model_pretrain_multiview_gated_fusion.py) |
| Residual | Original anchor plus gated lung/heart residuals | [Residual](../src/cxr_vlm/models/model_pretrain_multiview_residual_fusion.py) |
| Transformer | Two-layer cross-view self-attention then mean pooling | [Transformer](../src/cxr_vlm/models/model_pretrain_multiview_transformer_fusion.py) |
| Cross-attention | Two layers updating original query from lung/heart memory | [Cross-attention](../src/cxr_vlm/models/model_pretrain_multiview_cross_attention_fusion.py) |

Do not collapse these into a generic implementation during an organizational change.

## Classification and attribution

For finding `c`, use positive prompt `c` and negative prompt `no c`; the implementation special-cases `No finding` with negative `Finding`. The probability is the positive entry of `softmax([similarity_negative, similarity_positive] / temperature)`. The learned checkpoint temperature is used unless explicitly overridden.

The reported ALBEF ITC-margin attribution explains `(similarity_positive - similarity_negative) / temperature`, backpropagated to the penultimate visual self-attention block. It does not use the ITM head or multimodal XBERT during this inference pathway. Older patch-similarity, feature-CAM, rollout and ITM cross-attention extractors remain distinct historical methods.

Joint-fusion attribution follows gradients into all three visual branches. Sum their positive **unnormalized** maps, then maximum-normalize once per image. Do not explicitly reweight branch CAMs by saved gate values or interpret the result as fusion-layer attention.

BioViL-T exposes native local image–phrase similarity. CheXzero panels consume supplied external attribution. These are descriptive references with different spatial semantics, not automatically comparable implementations of one localization algorithm.

## Evaluation and statistical scope

The thesis reports held-out classification ROC-AUC and validation-thresholded F1. Validation labels also determine checkpoint selection and fitted ensemble weights. This is prompt-based zero-shot prediction with labeled development data, not an absence of validation supervision.

Pointing-game evaluation covers 309 Cardiomegaly and 111 PE positive-box image–finding pairs. Score saved image-resolution maps without extra smoothing, resizing or thresholding. Map pixel `(row, col)` to original coordinates `((col+.5)*width/map_width, (row+.5)*height/map_height)`; any target box, including its boundary, can yield a hit. Stable tied-maximum sampling uses seed 42; flat maps are misses; missing or invalid maps require coverage diagnostics.

Report Wilson 95% intervals, paired 10,000-image-bootstrap intervals, exact two-sided McNemar tests and Holm correction. Guidance and fusion define separate 12-comparison families across the two findings. These quantify peak alignment on positive-box cohorts, not lesion extent or false-positive localization on negative images.

The thesis qualitative series uses four positive radiographs. Larger extraction galleries are not the figure cohort or the quantitative evaluation denominator. Classification comparisons are descriptive point estimates, not significance-tested improvements.
