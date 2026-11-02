# Structural Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish the reviewed source/experiment hierarchy while preserving historical program behavior and invocations.

**Architecture:** Move Python implementations and configurations byte-for-byte. Retain legacy Python compatibility loaders and configuration symlinks, and provide one explicit canonical experiment launcher that restores historical import context. Keep notebooks, results, patient-data artifacts, bytecode, upstream vendor resources and image assets in place for this phase.

**Tech Stack:** Python 3.9-compatible standard-library launch/compatibility code; existing research dependencies; unittest.

**Spec:** [Reviewed architecture](../../repository_architecture.md), [methodology constraints](../../methodology.md), and `AGENTS.md` as amended for this approved structural phase.

## Global constraints

- Dedicated `refactor/thesis-repository` branch based on documentation commit `8d82b18`; no push, merge or main edits.
- No experiment body, architecture, preprocessing, loss, prompt, metric, dependency or config-value changes.
- Every migrated original is byte-identical at its canonical path; legacy paths remain usable.
- No deletion, untracking, result overwrite or publication clearance. Hold review-sensitive and resource-coupled material in place.
- A5/A6 stay preserved outside main thesis workflow; no invented MedKLIP/semi-supervised implementation.
- Existing API, tokenizer, data and GPU blockers remain documented.

## Review focus

- Preserve `__file__`, module identity, script arguments and sibling imports for historical execution.
- Avoid the new `src/` directory shadowing bare historical `src` helper imports.
- Relative config references must resolve through retained legacy config paths.
- Canonical launcher must accept only registered entry points and enforce repository-root working directory.
- No moved implementation/config bytes or held artifacts may change.

## Task 1: Characterize and test migration

**Files:** `tests/compatibility/{behavior_probe.py,reference.json,test_migration.py}`.

**Interfaces:** Fixed-seed pre-migration reference for synthetic image transform, caption, ViT/XBERT loss/gradients/state keys and five standalone fusion operators. Exact-byte validation covers FC and all other full model implementations without downloading weights.

- [x] Capture compact numerical reference from original implementation.
- [x] Run compatibility tests before implementation; expected failures identify absent launcher, migration manifest and src layout.
- [x] Add canonical import-path equivalence checks.

## Task 2: Apply mechanical moves and compatibility

**Files:** Canonical destinations from reviewed external migration map; `docs/audit/applied_migration.csv`; legacy Python loaders; legacy YAML symlinks; `src/{__init__.py,cxr_vlm/__init__.py}`; `tools/run_experiment.py`.

**Interfaces:** Manifest columns `legacy_path,canonical_path,original_sha256,kind`; kind is `entrypoint`, `module` or `config`. Launcher accepts canonical or legacy entrypoint path and forwards remaining arguments.

- [x] Move approved Python/config originals with `git mv`; verify every hash before moving; do not move held file categories.
- [x] Add legacy loaders that compile canonical bytes without changing original `__file__`/module context; preserve config lookup via relative symlinks.
- [x] Add src-name and canonical validation-helper compatibility imports.
- [x] Implement registered launcher and run all compatibility checks.

## Task 3: Document and review

**Files:** README, AGENTS, docs guides, registry canonical-path column, migration validation record and conservative `.gitignore` for new local caches/outputs only.

- [x] Update navigation to actual hierarchy and clearly identify retained legacy compatibility files/held artifacts.
- [x] Record exact checks, known failures and incomplete GPU/data coverage.
- [x] Verify Markdown links, registry references, unchanged original contents, shell syntax and whitespace.
- [x] Obtain independent review, address material issues and rerun affected checks.
- [x] Leave changes reviewable on the feature branch; no push/merge/publication.
