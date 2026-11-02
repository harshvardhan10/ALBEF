# Discrepancies and unresolved provenance

This record separates the thesis/reporting protocol, current code, historical evidence and remaining work. Do not change either the implementation or reported methodology silently to make them agree.

| ID | Topic | Evidence / remaining action |
|---|---|---|
| D01 | A0–A2 full data vs subset YAML | Researcher confirms full data. Commit `1b6a0a5` introduced subset selection; its predecessor never read the existing YAML key. Earlier full-data behavior is supported; final run configs/hashes remain missing. |
| D02 | Prepared manifest | Checked-in manifest has 215,394 records. Commit `452d57e` describes excluding 43 missing-mask images, matching thesis count 215,351 numerically. Generated common-valid manifests and per-run membership still need verification. |
| D03 | A3 support | Thesis §5.4.3 uses rectangular supports. Current YAML enables segmentation-derived heart prior added in `7562f6c`. Preserve as separate experiment identities; recover rectangular-prior run settings. |
| D04 | Guidance augmentation | Thesis §5.2.2 excludes random crop/flip. Checked-in guidance trainers use generic pretrain transform containing both. Recover exact final training revision/local modifications before claiming matched augmentation. |
| D05 | Checkpoint selection | Thesis §5.2.3 uses validation macro AUC. Guidance trainers internally save best training-loss checkpoint. Selection may have occurred externally; need logs and hashes, not an inference from `checkpoint_best.pth`. |
| D06 | Macro label set | Current dual-AUC config lists 15 stable labels, minimum support 20/20 and delta 0.0005; thesis describes 28 candidates excluding undefined AUCs. Recover executed label set and selection trace. |
| D07 | PE guidance and A3 sweeps | Named YAMLs primarily describe Cardiomegaly. A3.1/A3.2 revisions are recoverable from `ea61e1e`/`c9a5803`; PE-specific final configs and checkpoint identities remain absent. |
| D08 | Generic multi-view evaluator | Evaluator hardwires cross-attention model and retains generic output naming. Do not load other fusion checkpoints through it without an independently reviewed change. |
| D09 | Pointing-game execution | Notebook config contains seven guidance methods, blank checkpoint provenance and a reference to absent `analysis/localization_pointing_game.py`. Fusion-specific execution configuration/output is not checked in. Embedded analysis functions are present; absent module is a documentation gap, not proof the notebook cannot run. |
| D10 | Final results | Historical output paths are not sufficient to map thesis tables to checkpoints/protocols. Recover saved score/map/protocol/coverage/statistics bundles. |
| D11 | Legacy environment | Pillow 8.4 lacks `Image.Resampling` used by newer validation transforms. Some inherited task scripts require `ruamel_yaml` while requirements declare `ruamel.yaml`. Keep dependency fixes separate. |
| D12 | Launch and tokenizer | Bare `src` imports depend on launch directory. Shared helper hardcodes an HPC BERT cache with local-only loading. Record explicit launch conventions and verify assets. |
| D13 | HPC jobs | No standalone shell/SLURM file found in 494 distinct paths across available history. Commit messages mention jobs but do not recover original executable scripts. Any future templates must be labeled reconstructed. |
| D14 | External code | BioViL-T package/weight pins and CheXzero complete execution pipeline are missing. Do not invent repositories, revisions or checkpoint URLs. |
| D15 | Historical protocol correction | `40c3b56` records correcting test-derived F1 fitting and missing ensemble normalization. Treat preceding outputs as historical until their actual protocol is identified. |
| D16 | Commit-message ambiguity | `cc5daff` says validation used test data, but introduced config points to train. Follow config evidence; executed job still requires run logs. This message alone does not prove leakage. |

## Publication review

Tracked report-bearing manifests, patient-image previews/contact sheets, per-image results and embedded notebook outputs need redistribution review. Seventy bytecode files and tokenizer temporary files are removal candidates, not approved deletions. Java artifacts require license/version inventory. No current tracked model checkpoint file was found; all historical object contents have not been comprehensively scanned.

A limited secret-assignment heuristic found no matches. It is not a complete secret/history scan or a public-release clearance.

## Evidence labels

- **Implementation-verified:** current or historical source supports the statement.
- **Researcher-confirmed:** user supplied the experimental fact, such as full-data A0–A2.
- **Thesis-reported:** transcribed from the report, not independently regenerated.
- **Reproduced:** inputs, environment, command, checkpoint and output have been verified together. No full thesis run receives this label in this documentation phase.
