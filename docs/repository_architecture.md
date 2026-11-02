# Repository architecture and migration scope

Documentation was committed separately on `docs/thesis-research-guide` (`8d82b18`). The structural phase on `refactor/thesis-repository` moves 130 Python implementations and 45 configs byte-for-byte. A pure-move commit (`562bf51`) preserves rename history; compatibility files restore old paths. No historical material was deleted. The subsequent [opt-in asset IO revision](local_paths.md) preserves the two original helper snapshots; all model/method implementations remain unchanged. The original root README is preserved in [upstream documentation](upstream/README.original.txt).

## Research layout and remaining scope

```text
src/cxr_vlm/                       Reusable models, datasets, anatomy and evaluation
experiments/baseline/              Baseline entry points
experiments/anatomy_guidance/      A0–A4; A3.1/A3.2 remain explicit
experiments/single_view/           Original, lung and heart execution paths
experiments/late_fusion/           Score ensembles
experiments/joint_fusion/          Six separate architecture families
experiments/external_baselines/    Verified BioViL-T / CheXzero integration
experiments/evaluation/            Thesis classification/localization entry points
configs/historical/               Existing configs preserved verbatim
configs/reproduction/             Only provenance-verified thesis run settings
configs/local/                    Future ignored local overrides
tools/                            Preprocessing, inspection and diagnostics
hpc/                              Recovered jobs or labeled reconstructed templates
notebooks/                        Exploration, comparisons, evaluation and figures
results/historical/               Existing outputs preserved
results/thesis/                   Verified table/figure provenance
archive/non_thesis_experiments/    A5/A6, preserving dependencies
archive/exploratory_localization/  FROC/refinement history
archive/upstream_tasks/            Inherited non-thesis ALBEF tasks
docs/                             Research guides and provenance
```

Directory names do not imply missing baseline implementations exist. Do not create invented MedKLIP or semi-supervised workflows. Preserve source notices and state_dict identities when packaging modules.

## Ordered phases

1. Documentation and experiment registry: completed; source links now point to canonical implementations.
2. Provenance recovery: historical guidance configurations, target-specific runs, final manifests, checkpoints and evaluation outputs. Missing assets prevent verified reproduction claims, but not navigation improvements.
3. Mechanical moves: applied to Python/config files; see [applied manifest](audit/applied_migration.csv). No architecture or training-loop deduplication. Legacy source paths are compatibility loaders; legacy config paths are relative symlinks.
4. Separate environment/path changes: validate each stack and supported overrides; treat behavior corrections as separately reviewed work.
5. Equivalence checks: transforms, prompts, masks, parameter keys, representative forward/backward outputs, heatmap schemas/geometry and metrics.
6. Explicit artifact/publication review: preserve securely before any approved deletion/untracking/history rewrite. Never overwrite results/checkpoints.
7. Reproduce available thesis targets and report missing evidence. Merge/publication are separate actions.

The external review bundle retains the original proposed 433-file migration map with hashes. A [public source/config-only applied manifest](audit/applied_migration.csv) records the 175 completed moves. The original full 433-file map is not copied into this public-facing documentation because it inventories potentially restricted image/manifest paths. The compact experiment registry and commit lineage contain implementation references only.

## Compatibility and held categories

Use `python -B tools/run_experiment.py <canonical-or-legacy-path> [arguments]` from the checkout root. Only manifest-registered entry points are accepted. The launcher preserves legacy script directory, module and `__file__` context; it does not change working directory or reinterpret data arguments. Helper imports continue to use historical names deliberately. `src/__init__.py` bridges historical bare `src` imports that would otherwise be shadowed by the new layout.

For in-checkout reusable imports, set `PYTHONPATH="$PWD:$PWD/src"`. This phase does not create a separately installable distribution or promise canonical scripts support direct deep-path invocation. Use the launcher. Retained loaders preserve historical pickling/module names. Config symlinks preserve root-relative defaults and require symlink support; Linux was validated.

Seven output-free notebooks and 34 aggregate results are now organized, with old locations retained as symlinks; see [artifact organization](artifact_organization.md). Twenty-one output-bearing notebooks, two per-image prediction files, patient-data artifacts, bytecode, image assets, and RefCOCO/VQA Python/Java resource trees remain at their original paths. Their moves need working-directory/resource-path and redistribution review. Empty future directories in the conceptual tree are not represented as already implemented workflows.

## Acceptance criteria for later migration

Every historical file is accounted for; no destination collisions; old/new invocation behavior is documented; no unintended protected-file changes; no model/prompt/preprocessing/metric changes; each experiment retains its own ID/config/checkpoint mapping; results remain byte-identical; and unavailable GPU/data checks are explicitly pending.

Current source/config discrepancies must remain visible. Rewriting a historical configuration to match a thesis statement without run evidence is not a reproducibility fix.
