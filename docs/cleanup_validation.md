# Cleanup verification status

## Implemented and checked locally

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
- Specific approval and execution of the generated-file removal inventory and local-preserving data-manifest untracking. Nothing deleted.
- Final whole-cleanup validation and supervisor review acceptance. Main and source repositories remain unchanged; no push/merge/publish occurred.

Next: follow [cross_attention_validation.md](cross_attention_validation.md). A checkpoint smoke run validates technical loading/extraction only; it does not establish good localization or full training equivalence.
