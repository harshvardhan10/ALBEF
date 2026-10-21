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

## Still pending

- Actual ALBEF model construction/imports in the legacy environment, offline BERT cache, both real checkpoint loads, real VinDr preprocessing/maps, repeatability and baseline/refactored comparison.
- Exact training commit and saved resolved-config provenance; checkpoint paths/status are confirmed by author only.
- Model/trainer consolidation, broad historical launcher portability and focused training/DDP checks. These are not marked complete.
- Generated-file and data-manifest untracking was subsequently approved and completed with local copies preserved; see the model-candidate update below. Remaining hygiene is pending.
- Final whole-cleanup validation and supervisor review acceptance. Main and source repositories remain unchanged; no push/merge/publish occurred.

Next: follow [cross_attention_validation.md](cross_attention_validation.md). A checkpoint smoke run validates technical loading/extraction only; it does not establish good localization or full training equivalence.

## Model consolidation candidate (2026-10-22)

User-confirmed server evidence: original baseline and repeat runs completed for both checkpoint roles; both comparisons PASS, six cases per role and no all-zero CAM warnings. The saved training YAML matches the checked-in cross-attention config except comments. Training commit is unconfirmed. Evidence files were retained by the user on the server rather than inspected here.

The six model modules now delegate to one shared ALBEF model with separate FC/mean/gated/residual/Transformer/cross-attention implementations. All five extracted non-FC fusion class bodies match the frozen originals exactly. FC preserves its original separate normalization parameter names and always-on finite checking. Original model sources are retained as compressed independent test fixtures at the recorded baseline revision; no Git history was rewritten.

`OMP_NUM_THREADS=1 python -m unittest discover -s tests -q`: 24 tests pass. Five new test methods cover all six variants, testing strict state loading, exact initialization/queue values and parameter ordering, features, ITC/ITM/MLM losses, gradients and momentum freezing against frozen originals under tiny injected encoders. These tests do not prove real ViT/BERT/GPU/DDP behavior. The new test calls use torch.testing.assert_allclose for torch 1.8 compatibility; current local torch emits deprecation warnings. The Transformer also emits a current-torch nested-tensor optimization warning, not a numerical failure.

A single bounded independent review found no blocker for staging the server comparison. Remaining acceptance gaps include real encoder checkpoint loading, pretrained DeiT initialization, CUDA/AMP, nonzero dropout and distributed gather. Follow [cross_attention_refactor_check.md](cross_attention_refactor_check.md) before accepting the candidate or consolidating training loops. Existing root training scripts, historical configs and real data/results remain unchanged. The user explicitly approved untracking the 70 generated files and the real manifest. All 71 local files still exist and match their baseline hashes; they are now ignored. Older server checkouts must use the guide’s backup step before pulling the tracked removals.
