# Cleanup verification status

## Baseline slice — implemented and checked locally (2026-10-20)

- Read-only starting audit and local branch isolation at thesis-repository-cleanup, base 6320a2e93ce63bf2a77f3c716278c4379e9114f1.
- 431 original tracked files inventoried by hash. Only README.md modified; its original copy is docs/upstream-albef-readme.md. Original models, trainers, configs, data and results remain unchanged.
- 122 source files imported from the two recorded companion revisions: 117 byte-identical files plus five notebooks with source/cells preserved and generated outputs stripped. One generated server log excluded. Full source-to-destination hashes are in source_imports.json.
- New fusion-specific extractor, bounded two-checkpoint-role SLURM launcher, PNG overlay renderer and raw-map/score comparison utility.
- `OMP_NUM_THREADS=1 python -m unittest discover -s tests -v`: 19 tests pass, with no failures. The suite includes an extraction-to-artifact-to-comparison run using the actual unchanged anchored cross-attention classes with tiny synthetic vision/text encoders, and focused mask, gradient, strict state-key, schema, hook-cleanup, zero-map and output-preservation checks.
- Python 3.8 grammar parsing of all six new Python files succeeds. This is a syntax check, not execution on Python 3.8.
- `bash -n cluster/slurm/extract_cross_attention_fusion_heatmaps.slurm` succeeds. No job was submitted. Historical launch scripts are preserved, not asserted portable or runnable.
- One bounded independent review found no blocking issue in the validation slice. An optional incomplete-output comparison gap was fixed with a failing-then-passing regression test.

## Environment and warnings

Local Python 3.12 / torch 2.3.1+cu121 / torchvision 0.18.1 is different from server Python 3.8.20 / torch 1.8.1+cu111 / CUDA 11.1. The CPU suite reports Can't initialize NVML. Some synthetic maps are intentionally zero and produce explicit warnings. timm and transformers are absent locally; no dependencies were installed. No pre-existing automated research test suite was found; test_ddp_nccl.py is a cluster diagnostic, not run here.

## Historical milestones

The baseline and model-candidate sections below record checks at those stages. The final status at the end supersedes their pending-work notes. Main and companion source repositories remain untouched; authorized delivery is restricted to the cleanup branch.

## Model consolidation candidate (2026-10-22)

User-confirmed server evidence: original baseline and repeat runs completed for both checkpoint roles; both comparisons PASS, six cases per role and no all-zero CAM warnings. The saved training YAML matches the checked-in cross-attention config except comments. Training commit is unconfirmed. Evidence files were retained by the user on the server rather than inspected here.

The six model modules now delegate to one shared ALBEF model with separate FC/mean/gated/residual/Transformer/cross-attention implementations. All five extracted non-FC fusion class bodies match the frozen originals exactly. FC preserves its original separate normalization parameter names and always-on finite checking. Original model sources are retained as compressed independent test fixtures at the recorded baseline revision; no Git history was rewritten.

`OMP_NUM_THREADS=1 python -m unittest discover -s tests -q`: 24 tests pass. Five new test methods cover all six variants, testing strict state loading, exact initialization/queue values and parameter ordering, features, ITC/ITM/MLM losses, gradients and momentum freezing against frozen originals under tiny injected encoders. These tests do not prove real ViT/BERT/GPU/DDP behavior. The new test calls use torch.testing.assert_allclose for torch 1.8 compatibility; current local torch emits deprecation warnings. The Transformer also emits a current-torch nested-tensor optimization warning, not a numerical failure.

A single bounded independent review found no blocker for staging the server comparison. Remaining acceptance gaps include real encoder checkpoint loading, pretrained DeiT initialization, CUDA/AMP, nonzero dropout and distributed gather. Follow [cross_attention_refactor_check.md](cross_attention_refactor_check.md) before accepting the candidate or consolidating training loops. Existing root training scripts, historical configs and real data/results remain unchanged. The user explicitly approved untracking the 70 generated files and the real manifest. All 71 local files still exist and match their baseline hashes; they are now ignored. Older server checkouts must use the guide’s backup step before pulling the tracked removals.

## Final implementation and verification — 2026-10-22

**User-confirmed server acceptance:** both checkpoint-role baseline/refactored comparisons PASS after following all five refactor-guide steps; matched overlays reviewed and reported good. The earlier baseline/repeat comparisons also passed with no all-zero CAM warnings. The saved actual run config was compared by the author and differs from the historical repository config only in comments. Training commit remains `unconfirmed`. Server logs/manifests/comparison JSONs were not independently inspected here. This is technical compatibility acceptance, not a new localization-quality ranking.

**Implemented:** shared multiview training/dataset/checkpoint helpers and legacy CLI wrappers; separate anatomy/patch helper extraction; portable methodological/environment/smoke configs; portable explicit-input cluster launchers; common evaluation artifacts/roles/prompts/utilities; score/ensemble provenance; localization identifiers; exploratory FROC relocation with compatibility paths; dependency separation; notebook/legacy inventories and supervisor guide. Historical configs, companion source snapshots, original scientific notebooks and source repositories are retained. New intended-full-data presets are not evidence that historical training used all data.

**Local final checks:** 41 unittest tests pass (`OMP_NUM_THREADS=1 python -m unittest discover -s tests -q`). Coverage includes all six frozen-reference model variants, all six original multiview epoch functions with real optimizer steps, full-state resume, independent checkpoint-selection/deletion policies, configuration precedence, paired-record alignment, support/patch-loss distinctions, real A4 positional interpolation, legacy Pillow fallback, prompt/NPZ metadata and extraction/comparison/overlay behavior. Tiny encoder stand-ins avoid unavailable timm/transformers. One bounded training review found a missing lazy interpolation import in A4; a regression test reproduced NameError before the fix and passed afterward. No further concrete blocker was reported by that review.

Python 3.8 grammar checks pass for changed/new Python source (not Python 3.8 runtime execution). All three canonical SLURM scripts pass bash -n. The unified plus six compatibility training commands pass --help without the optional research stack. Unchanged older commands still require their original optional dependencies; an overly broad local CLI check encountered missing cv2 in the untouched Pretrain_A0_multiview_dual_auc.py and was narrowed to the seven consolidated entrypoints. No dependency was installed to hide this limitation.

Portable YAML parses and has no personal /home paths; intended-full-data presets have train_subset_size null. README/direct documentation links resolve. Anatomy/patch train/main function ASTs match the frozen original source. All 122 companion imports retain recorded destination hashes; four relocated FROC sources match recorded hashes. The relocated notebook dependency freeze matches original bytes. All notebooks parse; generated bytecode and the real manifest remain untracked. git diff --check passes.

**Not run:** full legacy ViT/BERT training, DDP/AMP/dropout runtime, anatomy/patch training loops, FROC/baseline execution and full environment installation. The model inference comparison is user-confirmed on the server; it does not prove these workflows. [Optional bounded server checks](cluster_workflows.md) retain the successful environment, use private inputs/new outputs and do not ask for retraining the completed thesis experiment. No jobs or messages were submitted by the agent. Supervisor review and final scientific conclusions remain the author's responsibility.

## Server training smoke check accepted — 2026-10-22

The author reports `PASS: smoke configuration and recovery checkpoint verified` after running the documented bounded one-GPU cross-attention training check, and confirms the logs are fine. Accept the smoke configuration and recovery-checkpoint checks as user-confirmed server evidence. Logs/checkpoint files were not independently inspected here. This covers the bounded training path and checkpoint writing; full training, two-GPU DDP and separate anatomy/patch workflows remain unverified. No further heatmap repetition or full retraining is required for the cleanup review package.
