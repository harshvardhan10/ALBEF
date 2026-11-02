# Portable Assets and Local Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make local tokenizer/model assets and historical config paths explicitly selectable without changing default experiment behavior.

**Architecture:** Add identical opt-in asset resolution to the two evaluation IO helpers, retaining original source snapshots and bounded-change provenance. Add a conservative local-config generator that accepts existing path fields only and never rewrites historical configs. Validate a separate CPU research environment for newer Pillow APIs; do not alter the existing legacy environment or claim to recover the final HPC environment.

**Tech Stack:** Existing Python 3.9/PyYAML/legacy ALBEF stack; standard-library unittest; isolated uv environment.

**Spec:** [Reproduction](../../../docs/reproduction.md), [discrepancies](../../../docs/discrepancies.md), `AGENTS.md`, and the user's instruction to continue on this branch with no PR/main merge.

## Global constraints

- Preserve models, loss, preprocessing, prompts, metrics, historical config values, data and results.
- Only two evaluation IO functions gain keyword-only asset overrides. Resolution: explicit keyword > explicitly set environment > historical default. Tokenizer local-only loading stays unchanged.
- Preserve original helper source snapshots and hashes. Existing migration records remain original-content evidence; record intentional current revisions separately.
- Generated local configs must not change hyperparameters, activate null fields or alter list cardinality. Only existing approved path fields may be rebound.
- Source config is immutable; output directory must be new and outside checkout or under ignored `configs/local/`.
- New validated dependency versions are candidate development evidence, not invented thesis environment pins.
- No restricted artifact removal, PR, branch switch or main merge.

## Review focus

- Default and explicit/environment precedence; empty or invalid asset overrides.
- Real local tokenization and checkpoint IO without private pretrained assets.
- AST equivalence outside the asset routing in both historical helper identities.
- Rejected hyperparameter/null/list-shape config changes and existing-output protection.
- Environment/version claims scoped to observed CPU checks; GPU and final run artifacts remain prerequisites.

## Task 1: Asset/config regressions

**Files:** `tests/compatibility/test_portability.py`.

- [x] Test historical defaults and keyword/environment precedence in both helpers.
- [x] Test real synthetic local tokenizer and checkpoint loading with model construction isolated.
- [x] Test bounded IO-only AST change against immutable snapshots.
- [x] Test config path binding, source immutability, disallowed fields, inactive fields, list shape and no overwrite.
- [x] Observe expected failures before implementation.

## Task 2: Explicit portability interfaces

**Files:** `src/cxr_vlm/evaluation/{legacy,modern}/src.py`, `archive/source_snapshots/evaluation/`, `tools/prepare_config.py`, original/current provenance CSVs and preservation tests.

- [x] Retain byte-exact originals and add `original_snapshot_path` for the two intentional revisions.
- [x] Add `_resolve_pretrained_sources` and keyword-only tokenizer/text-encoder paths; preserve default load flags and algorithm body.
- [x] Add `prepare_config(source, output_dir, overrides)` and CLI; record source/output hashes and path bindings locally.
- [x] Run focused and full compatibility checks.

## Priority update

The researcher confirmed that final run artifacts are unavailable here and prioritized easy access to model code for each main experiment. Environment-lock/reproduction work is deferred. An isolated CPU candidate installation was explored outside the checkout, but no candidate environment files are added or claimed as final thesis pins. Complete the already-written bounded IO/config work and prioritize direct full-source indexes/experiment guides.

## Task 3: Model-code navigation and guides

**Files:** `environments/`, relevant docs/agent instructions and validation records.

- [x] Inspect resources/pins before the isolated exploratory install (outside checkout).
- Deferred by priority update: validate/publish a complete CPU research environment.
- [x] Do not publish unvalidated candidate versions as recovered HPC pins.
- [x] Add model-code index and per-experiment guides linking to full implementation classes/methods.
- [x] Document bounded overrides, safe local config examples and residual blockers.
- [x] Obtain independent review, verify hashes/links/scope, commit and push current branch only.
