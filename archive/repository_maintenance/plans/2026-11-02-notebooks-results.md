# Notebook and Aggregate Result Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Organize the remaining analysis material on the current branch without changing notebook cells, stored outputs, historical result values or working-directory assumptions.

**Architecture:** Move only seven output-free notebooks and 34 aggregate CSV/JSON results byte-for-byte, retaining old paths as relative symlinks. Catalogue all 28 notebooks and 36 result files, including held output-bearing notebooks and per-image prediction files. Provide a standard-library context inspector that reports historical notebook execution paths without executing code.

**Tech Stack:** Python standard library, CSV/Markdown, existing unittest compatibility suite. No new dependency installs.

**Spec:** [Repository architecture](../../../docs/repository_architecture.md), [preservation rules](../../../AGENTS.md) and the user's instruction to continue only on this branch, without a PR to main.

## Global constraints

- Stay on `refactor/thesis-repository`; no PR, merge or direct main changes.
- No changes to notebook source cells, outputs, preprocessing, models, losses, prompts, metrics or result values.
- Hold 21 output-bearing notebooks and two per-image prediction files in place; no data cleanup or publication clearance.
- No execution of research notebooks, overwriting outputs, removal of historical experiments or new private-data blobs.
- Keep legacy notebook file parents and result paths usable. Canonical notebook location is for organization/viewing; historical execution context must be explicit.

## Review focus

- Output-free notebooks must also have no embedded attachments/image payloads.
- Kernel working directory matters independently of file contents and symlink targets.
- Aggregate results are historical, not newly reproduced thesis metrics.
- Compatibility symlinks must resolve to the exact registered targets.
- Held artifacts, original source/config files and all original content hashes must remain unchanged.

## Task 1: Regression checks

**Files:** `tests/compatibility/test_artifact_organization.py`.

- [x] Assert 28 notebook entries, seven safe moves and exact original content/legacy symlink targets.
- [x] Assert 36 result entries, 34 aggregate moves, two held prediction files and exact bytes.
- [x] Test context inspector on canonical/legacy paths and unknown paths, without notebook execution.
- [x] Observe expected failures before implementing missing catalogs/helper.

## Task 2: Mechanical organization

**Files:** `docs/audit/{notebook_catalog,result_catalog}.csv`, new canonical notebook/aggregate-result paths, legacy symlinks, `tools/notebook_context.py`.

- [x] Validate safe notebook/output/attachment and aggregate-result schemas before any move.
- [x] Apply exact `git mv` operations and preserve rename history in a separate commit.
- [x] Restore old paths with relative symlinks.
- [x] Add context inspector; do not execute or rewrite notebook cells.
- [x] Run compatibility tests and check all original contents.

## Task 3: Navigation and review

**Files:** `notebooks/README.md`, `results/README.md`, `docs/artifact_organization.md`, affected guides/agent instructions.

- [x] Document actual canonical paths, held categories, execution cwd and remaining provenance gaps.
- [x] Check links, catalogs, shell syntax, compatibility and hashes.
- [x] Obtain independent review and address material issues.
- [x] Commit current-branch changes; no PR or merge to main.
