# Repository architecture and migration scope

The current phase adds documentation and provenance on `docs/thesis-research-guide`. It does not apply the proposed source layout or delete historical material. The original root README is preserved in [upstream documentation](upstream/README.original.txt).

## Proposed next layout

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

1. Documentation and experiment registry: explain current paths, known evidence and gaps. This is the current deliverable.
2. Provenance recovery: historical guidance configurations, target-specific runs, final manifests, checkpoints and evaluation outputs. Missing assets prevent verified reproduction claims, but not navigation improvements.
3. Mechanical moves: one-to-one old/new map, small reviewable changes, `git mv`, import/reference updates and compatibility entry points. No architecture or training-loop deduplication in this pass.
4. Separate environment/path changes: validate each stack and supported overrides; treat behavior corrections as separately reviewed work.
5. Equivalence checks: transforms, prompts, masks, parameter keys, representative forward/backward outputs, heatmap schemas/geometry and metrics.
6. Explicit artifact/publication review: preserve securely before any approved deletion/untracking/history rewrite. Never overwrite results/checkpoints.
7. Reproduce available thesis targets and report missing evidence. Merge/publication are separate actions.

The external review bundle contains a proposed 433-file migration map with hashes. It is not copied into this public-facing documentation because it inventories potentially restricted image/manifest paths. The compact experiment registry and commit lineage contain implementation references only.

## Acceptance criteria for later migration

Every historical file is accounted for; no destination collisions; old/new invocation behavior is documented; no unintended protected-file changes; no model/prompt/preprocessing/metric changes; each experiment retains its own ID/config/checkpoint mapping; results remain byte-identical; and unavailable GPU/data checks are explicitly pending.

Current source/config discrepancies must remain visible. Rewriting a historical configuration to match a thesis statement without run evidence is not a reproducibility fix.
