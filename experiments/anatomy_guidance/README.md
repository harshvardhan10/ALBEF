# Anatomical-guidance experiments: A0–A4

**Start with the actual model:** [full ALBEF implementation](../../src/cxr_vlm/models/model_pretrain.py). These experiments share this architecture. Their differences live in the trainer's support/preservation losses and configuration, not in separate A1/A2/A3 model classes.

| Experiment | Model code | Method-specific code | Config evidence |
|---|---|---|---|
| A0 — No support penalty | [ALBEF](../../src/cxr_vlm/models/model_pretrain.py) | [trainer/loss integration](Pretrain_anatomy_prior.py) | [historical YAML](../../configs/historical/original_configs/Pretrain_anatomy_prior_A0.yaml) |
| A1 — Phrase-containing captions | [ALBEF](../../src/cxr_vlm/models/model_pretrain.py) | [trainer/loss integration](Pretrain_anatomy_prior.py) | [historical YAML](../../configs/historical/original_configs/Pretrain_anatomy_prior_A1.yaml) |
| A2 — Exact-positive/uncertainty weights | [ALBEF](../../src/cxr_vlm/models/model_pretrain.py) | [trainer/loss integration](Pretrain_anatomy_prior.py) | [historical YAML](../../configs/historical/original_configs/Pretrain_anatomy_prior_A2.yaml) |
| A3 — Positive-only; coefficient 0.01 | [ALBEF](../../src/cxr_vlm/models/model_pretrain.py) | [trainer/loss integration](Pretrain_anatomy_prior.py) | [historical YAML](../../configs/historical/original_configs/Pretrain_anatomy_prior_A3.yaml) |
| A3.1 — Positive-only; coefficient 0.005 | [ALBEF](../../src/cxr_vlm/models/model_pretrain.py) | [trainer/loss integration](Pretrain_anatomy_prior.py) | historical commit `ea61e1e` |
| A3.2 — Positive-only; coefficient 0.001 | [ALBEF](../../src/cxr_vlm/models/model_pretrain.py) | [trainer/loss integration](Pretrain_anatomy_prior.py) | historical commit `c9a5803` |
| A4 — Positive-only plus frozen-A0 similarity preservation | [ALBEF](../../src/cxr_vlm/models/model_pretrain.py) | [trainer/loss integration](Pretrain_anatomy_prior_A4.py) | [historical YAML](../../configs/historical/original_configs/Pretrain_anatomy_prior_A4.yaml) |

## Where to read the method

- [Caption-to-support weights](Pretrain_anatomy_prior.py#L50): exact equality, phrase presence and uncertainty handling.
- [Anatomical mask construction](Pretrain_anatomy_prior.py#L260): rectangular versus later external-prior path.
- [Pathology-token attention extraction](../../src/cxr_vlm/anatomy/attention_extract.py) and [token selection](../../src/cxr_vlm/anatomy/token_utils.py).
- [Outside-support loss](../../src/cxr_vlm/anatomy/losses.py#L41).
- [A0–A3 training objective integration](Pretrain_anatomy_prior.py#L285).
- [A4 classification-preservation loss](Pretrain_anatomy_prior_A4.py#L217) and [A4 training integration](Pretrain_anatomy_prior_A4.py#L319).
