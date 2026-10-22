# Source provenance

ALBEF is the review repository. The original source repositories remain unchanged.

| Source | Revision | Destination |
|---|---|---|
| harshvardhan10/ALBEF | 6320a2e93ce63bf2a77f3c716278c4379e9114f1 | existing root packages |
| harshvardhan10/thesis-codebase | 4b189fa90d8aa18be1ce22d003e908988cfa79b6 | cluster/historical, cluster/data_preparation, notebooks/data_preparation |
| harshvardhan10/CheXzero | 5f3997d8456fc2fe75742018557d79793cc546aa | baselines/chexzero |

[source_imports.json](source_imports.json) lists every companion file, SHA-256, destination and exclusion. 117 imports are byte-identical source snapshots. Five imported notebooks retain their cells/source but have generated outputs, execution counts and widget state stripped; both original and destination hashes are recorded. Only the generated server log is excluded from thesis-codebase. No dataset files or checkpoints have been imported. The CheXzero tokenizer vocabulary is retained.

CheXzero's LICENSE and NOTICE.md are retained in its directory. ALBEF's LICENSE and upstream attribution remain at the root. thesis-codebase has no declared license: these are the user's own supplied thesis scripts; no new license is asserted over them. Preserve third-party notices and resolve authorship before any redistribution.

Historical launch scripts retain their original paths, resources and notification directives as evidence. They are **not portable entry points** and must not be submitted unchanged. New portable launchers live in cluster/slurm, with no mail defaults. No jobs have been submitted.

## Cleanup transformations

Original model/trainer sources are frozen independently in tests/references/ at ALBEF revision 6320a2e. Model/fusion/dataset/checkpoint/support/patch helpers are extracted with compatibility entrypoints. Tests compare behavior against those original sources. New portable configs are labelled intended full-data presets; original run YAML is unchanged. Shared evaluation utility code replaces two originally identical copies; score arrays retain their schema while future shared-trainer outputs gain JSON metadata sidecars. FROC copies have relocation hashes and retained original command paths. Imported companion snapshots retain the source_imports.json hashes; the original notebook dependency freeze is relocated byte-for-byte. See legacy_inventory.md for retained material and cleanup_validation.md for verification limits.
