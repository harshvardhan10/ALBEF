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

- Continue on the current `refactor/thesis-repository` branch. Do not create a PR, merge to `main`/`master`, or switch branches unless the user explicitly asks. Inspect status first and preserve user changes.
- Cloud tasks already run in an isolated environment; use the existing checkout. Do not create a worktree unless the user explicitly requests one.
- No deletion, untracking, history rewriting or destructive cleanup without explicit user approval and recoverable preservation.
- Do not overwrite existing results or checkpoints. Select a new output directory and inspect the script's overwrite behavior.
- Do not commit patient images, report-bearing manifests, private datasets, credentials, model weights or machine-specific caches. Already-tracked historical material needs explicit review; ignore rules alone do not remove it.
- Keep mechanical moves/import updates separate from behavior fixes and dependency changes. Prefer `git mv` for approved moves and record old/new paths.

## Execution and dependencies

- Run historical commands from the checkout root. Scripts using bare `src` imports may require an explicit root `PYTHONPATH`; `python -m` is not interchangeable with direct execution.
- Read `docs/reproduction.md` before installing dependencies. Existing requirement files and `cog.yaml` describe different stacks, not one verified environment.
- Do not upgrade Transformers/timm/PyTorch or alter tokenization merely to make imports succeed. The local XBERT uses historical Transformers APIs.
- The legacy Pillow pin lacks `Image.Resampling`; newer validation-transform construction currently fails in that environment. The evaluation helpers retain the private tokenizer-cache default but now support explicit keyword/environment asset overrides; read `docs/local_paths.md`. Other known issues are not authorization for silent fixes.
- Use existing platform Git authentication, TLS verification and approved network routes. Never print credentials or request secret values in chat.

## Verification and claims

- For documentation: check links, cited source paths, CLI flags, CSV integrity and unchanged protected-file hashes. Do not run expensive GPU jobs for documentation-only edits.
- For future structural changes: verify old/new imports, dataset outputs, transforms, prompts, state_dict keys, representative forward/backward behavior, heatmap fields/geometry and metric outputs using fixed fixtures.
- Never change an assertion, metric, preprocessing or cohort to obtain a passing result. Missing assets/hardware are blockers to the affected check and must be reported.
- A module import or CLI help is not experimental validation. Distinguish static checks, CPU smoke tests, GPU execution and numerical thesis reproduction.
- Label values as thesis-reported until regenerated from identified data, config, environment, checkpoint and evaluation records. No fabricated dependency pins, URLs, checkpoint locations or results.
- Pointing-game guidance and fusion form separate 12-comparison test families. Do not pool them or tune extraction against test hit rates.

## Structural migration phase

The approved structural phase moves original Python/config files byte-for-byte and retains historical helper paths through compatibility loaders/config symlinks and former root entrypoint names through launcher aliases. No model/method-body or historical config-value edits are authorized. The two evaluation IO helpers have a separately recorded opt-in asset-resolution change; their original source snapshots and all other original functions are preserved. New local configs may rebind approved existing paths only via `tools/prepare_config.py`, never hyperparameters or inactive fields. Seven output-free notebooks and 34 aggregate result files are also organized byte-for-byte, with legacy symlinks. Output-bearing notebooks, per-image predictions, data, image assets and resource-coupled vendored tools remain in place. Use `tools/notebook_context.py` to inspect original notebook-parent context before running cells; a canonical file location does not establish the correct kernel cwd. Use `tools/run_experiment.py` for registered canonical entry points from the repository root; preserve historical helper imports and the documented wrapper execution context. Validate applied-migration hashes and numerical compatibility tests before completing changes.

## Current priority

The researcher confirmed final run artifacts are unavailable here. Prioritize `docs/model_code.md` and per-experiment guides so readers can reach each full model/method implementation. Do not block code navigation on unavailable checkpoints, invent per-guidance model subclasses, or broaden into speculative environment recovery. A0–A4 share ALBEF; six fusion architectures remain distinct.

## Branch cleanup

The researcher authorized clutter cleanup while leaving all notebooks untouched. Tracked Python bytecode and three tokenizer temporary files were removed after verified external backup; their original contents remain in Git history. Do not restore generated caches. Refactoring plans live under `archive/repository_maintenance/plans/`; phase validation records live under `docs/audit/history/`. Compatibility loaders remain intentional runtime support, not redundant research implementations.

Twenty-three former root command wrappers are grouped under `compatibility/entrypoints/`, recorded in the applied manifest’s `compatibility_path` column. Use the launcher with canonical or legacy names. Do not restore root duplicates. Keep imported helper wrappers (`models/`, `scripts/`, `utils.py`, etc.) and notebook cells unchanged. No migrated root command implementation uses `__file__`; reassess that constraint before changing wrapper execution context.
