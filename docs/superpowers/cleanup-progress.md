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
