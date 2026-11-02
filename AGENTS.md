# Repository instructions

## Purpose and scope

This is a Master's thesis research repository for anatomy-aware weak localization in chest X-ray vision–language models. Read `README.md`, `docs/experiments.md`, `docs/discrepancies.md` and the relevant source before making changes. The implementation and supplied thesis are evidence; flag disagreements rather than silently reconciling them.

## Protect research behavior

- Preserve model architectures, loss functions and coefficients, gradient/detach/freeze behavior, momentum pathways, queues, preprocessing, prompt templates, evaluation metrics and saved-map semantics unless a separate task explicitly authorizes changing them.
- Preserve all historical experiments and configurations. Similar names or similar implementations do not establish equivalence.
- A0–A2 reported runs used the full dataset; before commit `1b6a0a5`, the trainer ignored `train_subset_size`. Do not infer a subset run from its YAML alone.
- Main thesis guidance scope is A0–A4, including A3.1/A3.2 and separate Cardiomegaly/PE runs. A5/A6 remain historical and must not be deleted. MedKLIP implementation is outside the current scope. Do not invent a semi-supervised stage.
- Keep six fusion architectures distinct. Preserve original/lung/heart ordering and synchronized geometry. Fusion CAMs sum positive unnormalized branch attribution before a single maximum normalization.
- Preserve upstream license/copyright notices and document local changes.

## Git and files

- Work on a dedicated feature branch, never directly on `main` or `master`. Inspect status first and preserve user changes.
- Cloud tasks already run in an isolated environment; use the existing checkout. Do not create a worktree unless the user explicitly requests one.
- No deletion, untracking, history rewriting or destructive cleanup without explicit user approval and recoverable preservation.
- Do not overwrite existing results or checkpoints. Select a new output directory and inspect the script's overwrite behavior.
- Do not commit patient images, report-bearing manifests, private datasets, credentials, model weights or machine-specific caches. Already-tracked historical material needs explicit review; ignore rules alone do not remove it.
- Keep mechanical moves/import updates separate from behavior fixes and dependency changes. Prefer `git mv` for approved moves and record old/new paths.

## Execution and dependencies

- Run historical commands from the checkout root. Scripts using bare `src` imports may require an explicit root `PYTHONPATH`; `python -m` is not interchangeable with direct execution.
- Read `docs/reproduction.md` before installing dependencies. Existing requirement files and `cog.yaml` describe different stacks, not one verified environment.
- Do not upgrade Transformers/timm/PyTorch or alter tokenization merely to make imports succeed. The local XBERT uses historical Transformers APIs.
- The legacy Pillow pin lacks `Image.Resampling`; newer validation-transform construction currently fails in that environment. The shared model loader hardcodes a private tokenizer cache. These are known issues, not authorization for silent fixes.
- Use existing platform Git authentication, TLS verification and approved network routes. Never print credentials or request secret values in chat.

## Verification and claims

- For documentation: check links, cited source paths, CLI flags, CSV integrity and unchanged protected-file hashes. Do not run expensive GPU jobs for documentation-only edits.
- For future structural changes: verify old/new imports, dataset outputs, transforms, prompts, state_dict keys, representative forward/backward behavior, heatmap fields/geometry and metric outputs using fixed fixtures.
- Never change an assertion, metric, preprocessing or cohort to obtain a passing result. Missing assets/hardware are blockers to the affected check and must be reported.
- A module import or CLI help is not experimental validation. Distinguish static checks, CPU smoke tests, GPU execution and numerical thesis reproduction.
- Label values as thesis-reported until regenerated from identified data, config, environment, checkpoint and evaluation records. No fabricated dependency pins, URLs, checkpoint locations or results.
- Pointing-game guidance and fusion form separate 12-comparison test families. Do not pool them or tune extraction against test hit rates.

## Structural migration phase

The approved structural phase moves original Python/config files byte-for-byte and retains historical paths through compatibility loaders/config symlinks. No experiment-body or config-value edits are authorized. Notebooks, outputs, data, bytecode, image assets and resource-coupled vendored tools remain in place. Use `tools/run_experiment.py` for registered canonical entry points from the repository root; preserve legacy module and `__file__` context. Validate applied-migration hashes and numerical compatibility tests before completing changes.
