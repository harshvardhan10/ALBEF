# Structural migration validation

Branch: `refactor/thesis-repository`; documentation base: `8d82b18`; byte-for-byte move commit: `562bf51`. No push, merge or publication is implied.

## Applied scope

The [applied manifest](audit/applied_migration.csv) records 175 moves: 130 Python implementations and 45 configurations. Each original content SHA-256 is checked at its canonical destination. Historical Python paths are compatibility loaders; config paths are relative symlinks. The move commit contains zero inserted/deleted content lines and preserves rename history.

Notebooks, results, data, bytecode, image assets and resource-coupled RefCOCO/VQA tools remain at original paths. No restricted artifacts were copied into new source paths. The original README is preserved separately; all original research contents remain accounted for at a canonical or held path.

## Compatibility suite

Run from the root in the existing legacy CPU environment:

```bash
python -B -m unittest discover -s tests/compatibility -v
```

The tests first failed for the absent migration/launcher/layout. After implementation and final hardening, nine tests pass:

- Exact hashes and legacy-path availability for every applied move.
- Identical help output for four representative legacy/canonical commands.
- Rejection of unregistered entry points and wrong working directory.
- Historical bare `src` helper resolution in the new source layout.
- Canonical/legacy ViT output and state-key equivalence; canonical validation import.
- Preview tool's historical repository-root discovery.
- Historical executable mode/shebangs and direct execution of a representative executable script.
- Fixed-seed synthetic image/caption, small ViT/XBERT forward/loss/backward and state keys; mean, gated, residual, Transformer and cross-attention operator outputs/gradients.

The compact numerical fixture was captured before migration with seed 42 in the onboarding Python 3.9.25 / torch 1.8.0+cpu environment. Numeric comparisons use a declared 1e-6 relative-to-scale tolerance. It contains only synthetic tensors/summaries and parameter names, not dataset records or model weights. Exact source hashes also cover the full FC and ALBEF implementations, but full checkpoint-driven FC/ALBEF training/inference is not executed by this suite.

Existing torchvision interpolation warnings, vendored regex deprecation warnings and an original dataset-file ResourceWarning may appear; experiment source was not modified to hide them. New test files close their own files.

## Remaining limits

No final datasets, checkpoints, tokenizer cache, external model environment, GPU/DDP job, full-model training trajectory or thesis result was reproduced. Known Pillow API, historical YAML-module, hardcoded tokenizer and selection-provenance issues remain. Config compatibility requires symlink support. Reusable imports remain checkout-based; no installable wheel is supplied.

The legacy `test_ddp_nccl.py` is a CUDA/distributed diagnostic, not a CPU test-suite target. It remains available through its compatibility path and canonical diagnostic location.

## Independent review and final hardening

An independent reviewer verified all 433 original hashes, all 175 mappings, loader destinations/config symlinks, links and the initial eight-test suite. Review identified a documentation ambiguity (clarified) and optional manifest-test hardening (applied: exact 175-row scope and destination checks). Final verification additionally caught four lost executable modes/shebangs in compatibility loaders; a new regression test failed first, the original modes/shebangs were restored, and its targeted test passed. The complete nine-test suite passed in 21.989 seconds. Final static checks accounted for all 433 original contents, 175 exact moves, 104 local documentation links and nine Bash blocks.

Full-model GPU/data checks remain outside this review. The reviewer did not establish thesis-run provenance or numerical result reproduction.
