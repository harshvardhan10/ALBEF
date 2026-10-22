# Cleanup ledger — plan: docs/superpowers/plans/2026-10-20-approved-cleanup-plan.md

Approved 2026-10-20: ALBEF consolidation, ITC-margin ViT Grad-CAM, historical narrative/environment, no additional related-work repository. Cross-attention training completed; both named checkpoints are reported by user in /home/vault/iwi5/iwi5362h/outputs/A0_multiview_cross_attention_fusion_dual_auc. Exact training commit and resolved config remain unconfirmed.

Starting branch: thesis-repository-cleanup at 6320a2e93ce63bf2a77f3c716278c4379e9114f1. main is not checked out or modified. Companion repositories are read-only sources.

Pre-flight: Task 3 imports feed Task 2 catalogue. Task 4 baseline evidence gates Task 5 model consolidation; Task 5 shared model feeds Task 6 shared trainer. Task 7 metadata must preserve Task 4 extraction semantics. Task 8 runtime acceptance requires user evidence.

Ruling: Use this fresh branch checkout as isolation; no additional worktree is needed because no ALBEF checkout existed locally. No main checkout is touched.
Ruling: Keep historical launch scripts byte-identical under cluster/historical and add separate portable launchers, rather than automatically rewriting every machine-specific command. Cost: historical jobs require explicit migration, but scientific launch provenance is retained.
Ruling: Use unittest for focused CPU behavioral checks because pytest is absent; do not install GPU/research dependencies. Local torch 2.3.1 is not the server's historical torch 1.8.1.
Ruling: Preserve every original model/trainer until user-generated baseline evidence exists, as required by Task 4/5. No claim of completed structural refactoring is permitted.

- Task 1: complete for baseline inventory. 431 starting tracked files inventoried; 122 companion files imported (117 byte-identical plus five output-stripped notebooks) and one generated log excluded. Only original README changed, with original copy preserved.
- Task 2: review map/catalogue/provenance/checkpoint/reproduction/limitations documentation created. Checkpoint/result claims remain tied to author confirmation, not new runtime evidence.
- Task 3: companion snapshots imported; historical launcher migration remains pending.
- Task 4: extractor, bounded dual-role SLURM launcher, overlay renderer and raw-map comparator implemented. 19 CPU tests pass, including real anchored-fusion classes with tiny synthetic encoders. Actual server baseline, repeatability and legacy environment acceptance pending user execution.
- Tasks 5–6: pending baseline evidence.
- Task 7: destructive hygiene pending concrete inventory approval; runtime dependency files remain historical evidence.
- Task 8: pending remaining work and user validation.

Review: one bounded fresh-context review found no blockers in extractor/comparator/tests/launcher/guide. Optional incomplete-schema concern addressed with a regression test observed failing before the fix. Full server/model-assembly coverage remains unavailable.

Ruling: Do not install missing timm/transformers/pytest locally. The supplied CPU torch 2.3.1/torchvision 0.18.1 can exercise tensor and file behavior; full torch1.8.1 checkpoint tests remain on the server. Cost: actual legacy imports/weights are not locally certified.

Verification notes: TDD failures observed for absent extractor/comparator/renderer, missing manifest image_res in integration, and identically incomplete comparison payloads. Final 19-test suite passes. CPU environment emits Can't initialize NVML; synthetic zero-CAM fixtures intentionally exercise warnings. No suppressed scientific failures.

Next action: user runs fixed-case baseline twice for both checkpoint roles using docs/cross_attention_validation.md, supplies training commit/resolved-config location and logs/manifests/comparison evidence. Continue independent launcher/documentation migration while available; do not begin fusion/model consolidation before baseline acceptance. 70 tracked generated Python artifacts are listed in docs/proposed_generated_removals.md for specific approval. No destructive hygiene performed.

Import correction: final notebook inspection found generated outputs in five companion notebooks. Stripped only outputs/execution counts/widget state in these newly imported copies, as the approved plan requires; source cells retained and source/destination hashes recorded. Original ALBEF notebooks and source repositories remain unchanged.

## 2026-10-22 — baseline accepted and model consolidation started

User reports both baseline/repeat comparisons PASS, both checkpoint roles completed, six overlays per role, and no all-zero CAM warnings. Evidence files remain on the server and were not independently inspected. Saved training config matches the repository config except comments. Training commit remains unconfirmed. Baseline: /home/vault/iwi5/iwi5362h/outputs/cleanup_validation/baseline_20261022_003021.

Ruling: accept the user's explicit report as the baseline gate, as agreed in chat; retain the distinction between user-confirmed and independently inspected evidence. Cost: manifest/log details cannot be audited locally.
Ruling: reconcile local 7468e42 and API-created 639fc12 via a content-identical merge, without rewriting either history. Cost: an extra local merge commit; main untouched.
Ruling: FC retains top-level view_fusion_norm and separate momentum normalization rather than nesting them in the extracted fusion module. This preserves historical checkpoint keys and parameter order. Five other fusion classes are extracted without internal changes.
Ruling: shared objective uses the stable mean-model body; FC retains unconditional finite checks. Only FC's exception wording/diagnostic scalar formatting changes; finite-data objective calculations are compared against the frozen FC source. Cost: legacy error messages differ, not scientific computations.

Model work: six historical model modules become compatibility subclasses; models/model_pretrain_multiview.py owns common objective/queue/momentum logic; models/fusion owns architecture-specific math. Tests retain compressed independent original sources and compare all six model variants with tiny synthetic encoders. A failing FC feature comparison exposed a missing normalization dispatch; fixed before acceptance. 24 focused CPU tests currently pass (5 new test methods cover all six variants), actual server candidate comparison still required.

Do not accept Task 5 as runtime-validated until both fixed-case server baseline/candidate comparisons pass. Task 6 full trainer consolidation remains behind that acceptance gate. Broad historical launcher migration, anatomy/patch helper extraction, immutable configs and metadata integration remain pending. No model-training changes or generated-file removal has yet occurred in this slice.

Independent model review: no blocker for staging server comparison. Five extracted fusion classes AST-identical; all six state/RNG-order comparisons pass under synthetic encoders. Actual encoder initialization, GPU/AMP/dropout/DDP coverage remains pending. No training-script changes were made.

Training audit finding for Task 6: mean/gated/residual checkpoint payloads currently copy the Transformer architecture string despite selecting the correct model imports. Correct that metadata when consolidating trainers; do not retrospectively relabel historical checkpoints as newly verified runs. Gated/mean/residual also guard last-checkpoint removal on both best files existing; FC/Transformer/cross-attention have a different cleanup policy. Do not silently unify these behaviors.

2026-10-22 user approved stopping tracking all 70 listed generated .pyc files and data/mimic_cxr.json, preserving local copies. Verified every local file against its baseline hash before git rm --cached; all 71 still exist afterward. Ignore rule added for the real manifest. Server guide backs up these files before pulling, since remote tracked removals otherwise remove working copies. Source/history unchanged. Task 7 generated-artifact/manifest untracking complete; remaining hygiene/config/dependency work still pending.

Manifest hygiene follow-up: added data/mimic_cxr.example.json with synthetic image/caption fields confirmed against dataset/caption_dataset.py, and documented the imported preparation CLI. Server backup includes all tracked .pyc paths, including older generated files outside __pycache__.

## 2026-10-22 — model accepted; remaining implementation delivered

User reports both checkpoint-role candidate/baseline comparisons PASS, all five server steps completed, matched overlays good. Accept Task 5's agreed server gate. Historical training commit remains unconfirmed; actual saved config compared by user. Evidence files remain server-side and are not independently audited.

Tasks 1–5: implemented; user baseline and model-refactor acceptance complete. Task 6: shared trainer/dataset/checkpoint extraction, old CLI wrappers, anatomy/patch helpers and portable configs implemented. Task 7: common evaluation artifacts/roles/prompts, provenance sidecars, localization identifiers, exploratory FROC wrappers, dependency split and notebook/legacy indexes implemented; approved untracking already complete. Task 8: supervisor package and verification documentation delivered. Full legacy training/DDP acceptance remains unperformed, explicitly optional/bounded in docs/cluster_workflows.md; no full runtime-equivalence claim.

Ruling: preserve mean/gated/residual's guarded last-checkpoint retention versus FC/Transformer/cross-attention's original cleanup policy. Preserve A5 clamp-one versus A6 epsilon weighted-loss denominators and A6 positive-only support default. Correct previously copied Transformer architecture labels for future runs only. Standardize console diagnostics without changing loss/optimizer/scheduler math. Keep original anatomy/patch loops and historical YAML unchanged. Keep root packages and wrappers rather than broad src migration.

Ruling: archived launchers remain byte-identical; new explicit-input training/Python launchers cover consolidated workflows. External installations referenced by historical jobs are archival dependencies, not newly supplied code. Original ALBEF scientific notebooks/outputs remain evidence; only the previously recorded imported preparation outputs were stripped. Root runtime dependencies now select recorded legacy pins; original notebook freeze is preserved separately. No environment upgrade/installation, job submission or account connection occurred.

Verification: 41 CPU tests pass, including all six frozen-reference multiview epoch optimizer comparisons, strict model comparisons, resume and differentiated helper behavior. Python3.8 grammar, seven consolidated CLI help commands, three canonical shell scripts, portable config/doc links, unchanged anatomy/patch loop ASTs, all companion/FROC hashes and dependency snapshot verified. Missing cv2 in an untouched older trainer's eager imports remains an environment limitation, not a consolidated CLI failure. One bounded review exposed A4 missing interpolation import; failing-then-passing real checkpoint/interpolation regression added. Other runtime limits are in docs/cleanup_validation.md.

Delivery ruling: retain the existing checkout and push only thesis-repository-cleanup under the user's prior explicit authorization. Do not merge or update main, create a PR, rewrite history or delete any additional files. Final branch tree is verified against the uploaded Git tree before moving the remote ref.
