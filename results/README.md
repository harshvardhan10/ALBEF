# Historical results

These outputs are preserved research history, not verified generating records for the final thesis tables. No values were recalculated or rewritten. See [thesis result provenance](../docs/results_provenance.md).

## Aggregate records

- [FROC curves and summary](historical/zero_shot_vindr_results/froc_top_1): 28 curves plus one summary, preserved byte-for-byte. FROC is a historical localization protocol, not the thesis pointing-game endpoint.
- [Classification summaries](historical/vindr_zero_shot_results_a40): five historical JSON summaries. Their protocol/checkpoint identities must be recovered before final-result claims.

Old paths remain relative symlinks so original references still resolve. Absolute HPC paths inside stored summaries are unchanged for provenance, not portability guarantees. No original result was overwritten or discarded.

## Held per-image files

The two CSV/NPZ prediction files under `zero_shot_vindr_results/classification/` remain at their original locations. They contain image-level scores/labels and need dataset-license and identifier review; they are not copied into new result directories.

The [result catalog](../docs/audit/result_catalog.csv) accounts for all 36 files: 34 aggregate moves and two held prediction files, with original hashes. Symlink compatibility requires filesystem/Git support.

Write new experiment outputs to a fresh directory, preferably an ignored local location such as `results/runs/`. Do not overwrite these historical outputs. No final `results/thesis/` payload is invented from the PDF.
