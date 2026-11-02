# Documentation-phase validation (before source migration)

This is the historical record for documentation commit `8d82b18`; see [migration validation](migration_validation.md) for the subsequent structural phase.

Scope: documentation and provenance only, on branch `docs/thesis-research-guide`, based on `b63533571fe9ca2fb1dd4f1f5978d92eddb9f24b`. Source layout and research behavior are unchanged.

## Current-instance checks

The existing onboarding environment uses Python 3.9.25, torch 1.8.0+cpu, torchvision 0.9.0+cpu and the installed legacy requirements. `uv pip check` reports all 33 installed packages compatible. This is dependency metadata consistency, not API coverage of every research script.

Commands below ran from the checkout root with `PYTHONDONTWRITEBYTECODE=1`, `python -B` and the checkout root prepended to `PYTHONPATH`. Every help check exited 0 and produced usage text; no model was instantiated and no dataset/checkpoint was evaluated.

| Command | Result |
|---|---|
| `python -B Pretrain.py --help` | Exit 0; usage printed |
| `python -B Pretrain_anatomy_prior.py --help` | Exit 0; usage printed |
| `python -B Pretrain_anatomy_prior_A4.py --help` | Exit 0; usage printed |
| `python -B Pretrain_A0_multiview_dual_auc.py --help` | Exit 0; usage printed |
| `python -B Pretrain_A0_multiview_cross_attention_fusion_dual_auc.py --help` | Exit 0; usage printed |
| `python -B scripts/evaluate_zero_shot_f1_from_validation.py --help` | Exit 0; usage printed |
| `python -B scripts_new/zero_shot_eval_vindr_pos_neg_bare_prompts.py --help` | Exit 0; usage printed |
| `python -B scripts_new/learn_dual_checkpoint_ensemble_weights.py --help` | Exit 0; usage printed |
| `python -B scripts_new/extract_itc_margin_gradcam_heatmaps.py --help` | Exit 0; usage printed |

Documentation checks verify local link targets, shell syntax of documented Bash blocks, registry implementation paths and referenced lineage commits. Hash comparison covers all 433 original tracked files: only the intended root README edit differs. Its predecessor is preserved byte-for-byte in `docs/upstream/README.original.txt`. No original source, config, dependency declaration, notebook, result, image or manifest is changed.

## Known limits

- CPU model-component smoke validation was performed during onboarding; this documentation phase did not add or run a portable numerical research test suite.
- CLI help does not exercise newer validation transforms; Pillow `Image.Resampling` remains a known incompatibility in this legacy environment.
- Missing private tokenizer paths, prepared manifests, GPU hardware, final checkpoints and run provenance remain unresolved.
- Dataset-dependent command templates are unexecuted; final thesis numbers are thesis-reported, not reproduced.
- No source migration, new environment lock, artifact cleanup, push, merge or publication has occurred.

The original `test_ddp_nccl.py` is a distributed CUDA diagnostic, not an applicable documentation test. No GPU job or notebook was run for this phase.

## Independent review

A separate reviewer read the documentation and registry against the thesis/source evidence and found no actionable issues. The reviewer independently confirmed protected-file hashes, exact upstream README preservation, local links, Bash syntax, registry/CSV references, lineage commits and whitespace checks. CLI checks were not independently repeated; GPU/private-asset execution, final checkpoint selection and numerical reproduction remain outside this documentation review. Changes are left uncommitted for review.
