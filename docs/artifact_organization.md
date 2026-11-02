# Notebook and result organization

This phase continues on `refactor/thesis-repository` without a PR or merge to main. The prior Python/config compatibility layer and all model/experiment code are unchanged.

## Applied scope

| Material | Total cataloged | Moved unchanged | Held in place |
|---|---:|---:|---:|
| Notebooks | 28 | 7 output-free | 21 with stored outputs |
| Result files | 36 | 34 aggregate CSV/JSON | 2 per-image predictions |

The [notebook catalog](audit/notebook_catalog.csv) and [result catalog](audit/result_catalog.csv) record every original path, canonical path and original SHA-256. Old moved paths are relative symlinks to canonical originals. Moves use a separate pure-rename commit, preserving history. No notebook cells/outputs, result values, model code, preprocessing, prompts or metrics are changed.

Output-free notebooks were also checked for embedded attachments/data-image payloads. The aggregate CSV schemas contain curve/summary metrics, not image-ID rows; the JSON schemas contain aggregate classification records, not raw predictions. Existing report/data/previews, bytecode and vendor resources remain held. This is a scope classification, not a legal/publication clearance.

## Kernel cwd is separate from notebook location

Use the standard-library inspector:

```bash
python -B tools/notebook_context.py notebooks/evaluation/localization_pointing_game.ipynb
```

It reports a canonical viewing path, the legacy execution alias and the original file-parent directory. The directory is derived from historical file location, not proven final HPC launch provenance. It does not execute notebook cells or set kernel state. Open the legacy alias and verify cwd before execution. If the notebook server resolves symlinks or starts kernels elsewhere, establish the reported original-parent context first; do not assume opening the canonical path preserves it.

The original root `visualization.ipynb` retains a repository-root parent context, unlike the 27 notebooks originally under `notebooks/`. Older extraction notebooks use `os.chdir("..")`, and other notebooks use `../results` or local output folders. No notebook is rewritten to hide these assumptions.

## Checks and limits

Four new standard-library tests first failed for missing catalogs/inspector and now pass: all notebook bytes and safe move scope; all result bytes and held predictions; matching canonical/legacy context inspection (including root notebook); and unknown-path rejection. Existing numerical/launcher compatibility tests also remain applicable.

No research notebook, kernel, model inference, dataset, checkpoint or new scientific evaluation was executed in this phase. Stored outputs were not cleared. The original thesis provenance and environment gaps remain. Links, source hashes, symlink targets, shell syntax and full compatibility checks are recorded after final review.

## Final verification

All 13 compatibility tests passed in 22.784 seconds, including the four new artifact/context tests and nine existing source/launcher/numerical checks. Static checks accounted for all 433 original contents, 148 local documentation links and 12 Bash blocks. An independent reviewer checked all 64 notebook/result catalog hashes against the prior branch base, all 41 relative symlink targets, all stored-output counts and the four new tests; no blocking issue was found. One stale archival-description sentence was corrected.

The independent review did not execute research notebooks or validate live kernel/symlink handling, private-asset pipelines, scientific reproduction or redistribution rights. Continue only on the current branch; no PR or merge to main is part of this work.
