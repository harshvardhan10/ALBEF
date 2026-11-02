# Thesis Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make the current thesis repository navigable and document verified methodology, historical evidence and reproduction gaps without changing experiment behavior.

**Architecture:** Documentation-only first phase of the reviewed restructuring proposal. Preserve the prior README verbatim, add thesis guides and a compact experiment/lineage registry, and leave source paths intact.

**Tech Stack:** Markdown, CSV; Python standard-library validation outside the repository.

**Spec:** [Architecture and scope](../../../docs/repository_architecture.md), [experiment mapping](../../../docs/experiments.md) and [discrepancies](../../../docs/discrepancies.md), implementing the user's thesis/audit request and corrections.

## Global Constraints

- Dedicated branch; no direct main changes.
- No model, loss, preprocessing, prompt, metric, dependency, historical config, result or checkpoint changes.
- Preserve A5/A6 as non-thesis history; exclude MedKLIP from implementation scope.
- A0–A2 reported runs use full datasets; retain distinction between source evidence and final run proof.
- No deletion/untracking or overwriting research outputs. No new private assets, report PDF, patient identifiers or credentials.
- Documentation-only work uses link/reference/content validation rather than new production tests or GPU jobs.

## Review Focus

- Reported numerical values must cite thesis tables and be labeled not independently reproduced.
- A proposed architecture must not be described as the actual source layout.
- CLI/help checks must not imply that dataset-dependent templates or GPU experiments passed.
- Historical config changes must not become falsely verified final run provenance.
- A5/A6 exclusion must not become deletion or lost dependency history.

## Task 1: Thesis navigation and reference preservation

**Files:** Modify `README.md`; create `docs/upstream/README.original.txt`, `docs/methodology.md`, `docs/experiments.md`, `docs/repository_architecture.md`.

**Interfaces:** Consume current source paths, supplied thesis and reviewed audit; produce navigation used by all subsequent guides.

- [x] Preserve prior README byte-for-byte.
- [x] Write current-layout navigation and thesis-stage mapping.
- [x] Check local Markdown links and model/trainer/config targets; compare preserved README to base commit.

## Task 2: Reproduction, provenance and future-agent constraints

**Files:** Create `AGENTS.md`, `docs/reproduction.md`, `docs/data_access.md`, `docs/external_dependencies.md`, `docs/discrepancies.md`, `docs/results_provenance.md`, `docs/audit/experiment_registry.csv`, `docs/audit/historical_lineage.csv`.

**Interfaces:** Consume Task 1 source map; produce explicit environment/asset prerequisites and read-only registry references.

- [x] Write known requirements and incomplete evidence without inventing versions/URLs.
- [x] Preserve current configs and clearly separate unexecuted templates from CLI checks.
- [x] Validate all registry implementation paths and lineage commits; check template flags against parser definitions.

## Task 3: Verification and independent review

**Files:** Create `docs/validation.md`; update this plan's checkboxes after completion.

**Interfaces:** Consume Tasks 1–2 documentation and baseline audit hashes; produce exact validation outcomes for review.

- [x] Run documentation link/reference checks and documented CLI help commands in the existing legacy environment.
- [x] Verify every original tracked file except the intended README edit against baseline hashes.
- [x] Run `git diff --check`; inspect changed/untracked paths for scope and sensitive assets.
- [x] Obtain an independent documentation/provenance review, address material issues and rerun affected checks.
- [x] Leave changes reviewable on the dedicated branch; no push, merge or publication.
