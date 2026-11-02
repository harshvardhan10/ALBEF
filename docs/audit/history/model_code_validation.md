# Model-code access and bounded portability validation

This phase prioritizes access to the full implementation of each main experiment, as requested by the researcher. Final run configs, manifests, checkpoints and evaluation outputs are unavailable here; no numerical thesis reproduction is claimed.

## Navigation

[The model-code index](../../model_code.md) and guides beside the baseline, guidance, single-view, six fusion and late-fusion entry points link to actual implementation files rather than compatibility loaders. Static checks verify seven full ALBEF model sources, all 38 class/function line anchors, 351 local documentation links and 14 Bash blocks.

A0–A4/A3.1/A3.2 share an ALBEF architecture, with support/preservation logic in the trainers and anatomy helpers. Original/lung/heart models differ in input construction. Six fusion implementations remain distinct. Late fusion is score combination, not an invented neural-network architecture.

## Preserved code and optional IO revisions

All 433 original contents remain available at canonical/held paths or original snapshots. Two evaluation IO helpers have explicitly recorded, default-preserving asset overrides; their original versions and hashes are retained. No model, training-method, loss, preprocessing, prompt, metric, historical config, notebook or result content is changed by this feature.

The original-content test only permits snapshot exceptions for those two helpers and checks their current hashes against the approved change record. Portability tests also compare every old helper function's AST, normalizing only the added asset resolution/signature and routed asset arguments. Other original function bodies must remain identical.

## Runtime checks

The full compatibility suite runs in the existing Python 3.9.25 / torch 1.8.0+cpu legacy environment:

```bash
python -B -m unittest discover -s tests/compatibility -v
```

All 21 tests passed in 22.555 seconds. The suite includes the earlier numerical/launch/artifact checks plus eight portability tests. These cover both helpers' defaults and override precedence, actual synthetic local-vocabulary tokenization and checkpoint IO, source-boundary checks, config path binding/source preservation, rejection of semantic/inactive/list changes, duplicate overrides, dangling output symlinks and YAML-alias isolation.

The unavailable pretrained model constructor alone is isolated in the asset IO test; real tokenizer loading/tokenization, checkpoint deserialization, state loading, device selection and eval mode execute. This is not full pretrained ALBEF inference. The config CLI was also exercised on a fresh synthetic source/output and correctly rejected a repeat output directory.

The original legacy environment's installed package freeze remained identical before/after an exploratory isolated candidate install. No candidate environment or new dependency lock is published following the priority change to model-code access. Original dependency declarations remain unchanged.

## Review limits

Independent read-only review checked model/method mappings, direct implementation links and definition anchors, all original/current/snapshot hashes, default/override semantics and config safeguards. It did not rerun the tests or execute private assets, notebooks, GPU jobs or final thesis reproduction. No PR or merge to main is part of this phase.
