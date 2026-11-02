# Results and provenance

All numerical values here are **thesis-reported**, not independently reproduced in this documentation phase. Historical files under `results/` and `vindr_zero_shot_results_a40/` have not been established as the generating records for these tables.

## Selected reported outcomes

| Table | Quantity | Value |
|---|---|---|
| 6.1 | Original A0 Cardiomegaly ROC-AUC / F1 | 0.8764 / 0.4946 |
| 6.1 | Original A0 PE ROC-AUC / F1 | 0.9788 / 0.6333 |
| 6.3 | A1 Cardiomegaly ROC-AUC | 0.9043 |
| 6.3 | A3.2 Cardiomegaly / PE F1 | 0.5392 / 0.6900 |
| 6.6 | Validation-tuned late-fusion PE F1 | 0.7333 |
| 6.7 | Anchored cross-attention Cardiomegaly ROC-AUC / F1 | 0.9427 / 0.6125 |
| 6.8 | Gated fusion Cardiomegaly pointing rate | 82.20% |
| 6.8 | Anchored cross-attention PE pointing rate | 17.12% |

Anatomy and fusion have pathology-dependent effects; classification ranking does not establish localization quality. Classification differences are descriptive. Pointing-game paired tests apply to positive-box cohorts and defined comparison families; they are not full-population localization guarantees or pairwise rankings against every model.

## Table-to-artifact recovery map

| Thesis target | Implementation candidates | Required generating artifacts |
|---|---|---|
| Table 6.1 single views | Bare-prompt evaluator; validation-F1 evaluator | Three selected checkpoints, validation/test predictions, label IDs, calibration thresholds |
| Table 6.3 guidance | A0–A3/A4 trainers; target-specific score evaluation | Target-specific configs/checkpoints; A3 coefficient and prior identity; selection records |
| Tables 6.4–6.5 guidance localization | Pointing-game notebook | Protocol, checkpoint provenance, full cohort/coverage, per-image outcomes and paired statistics |
| Table 6.6 late fusion | Normalized ensemble evaluator; fitted-weight script | Three input prediction sets; validation z-score statistics, weights and thresholds |
| Table 6.7 joint fusion | Six model/trainer families; per-family evaluation commands | Correct architecture/checkpoint pairing, final score files, calibration and selection |
| Table 6.8 fusion localization | Fusion extractors; generic embedded pointing functions | Fusion-specific execution config, composite maps, coverage and per-image outcomes |
| Figures 6.1–6.16 | External/guidance/fusion comparison notebooks | Exact four source radiographs, annotations, map fields and figure-generation settings |

No verified public thesis URL/DOI is available. The privately supplied report is a reference, not a newly committed artifact.

## Run manifest requirements

For each reproduced target, record code revision; exact command; environment/Python/CUDA/GPU; seed; config hash; training/validation/test manifest hashes; model/tokenizer assets; initial and selected checkpoint hashes; selection label set/rule; validation calibration; score/mapping schemas; output paths/hashes; and completed check results.

For localization also record attribution target and layer, interpolation/coordinate transform, raw/normalized field names, cohort membership, missing/flat/tied-map diagnostics and statistical family. Distinguish guidance's single-view attribution from fusion's branch composite and BioViL-T native similarity.

## Existing output handling

Preserve original outputs byte-for-byte. Use a new run directory for every validation/reproduction attempt and inspect overwrite behavior before launching. Do not clear notebook outputs or rewrite result paths in stored files as part of a mechanical move.

Commit `40c3b56` records a protocol correction for validation-only F1 fitting and ensemble normalization. Earlier results require identification before being labeled final. Zero-ground-truth classes in historical FROC files are not automatically comparable to the thesis pointing-game cohorts.

Status labels: **historical**, **superseded when established**, **thesis-reported**, and **reproduced**. A filename or successful script exit does not promote a result to reproduced without checking target, coverage and provenance.
