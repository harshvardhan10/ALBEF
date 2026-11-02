# Optional local asset and configuration paths

The main priority is [direct access to model code](model_code.md). Final run configs, manifests, checkpoints and evaluation outputs are unavailable here; these helpers improve path selection without claiming to recover those artifacts or reproduce thesis results.

## Evaluation tokenizer/model assets

The two historical evaluation helpers now accept:

```python
build_model_and_tokenizer(
    config_path,
    ckpt_path,
    device='cuda',
    tokenizer_path='/absolute/path/to/the-verified-tokenizer',
    text_encoder='/absolute/path/to/the-verified-encoder',
)
```

These are keyword-only additions. Resolution is explicit keyword, then `ALBEF_TOKENIZER_PATH` / `ALBEF_TEXT_ENCODER` when set, then the original defaults. With neither override set, the historical local tokenizer cache and `bert-base-uncased` encoder selection remain unchanged. Tokenizer `local_files_only=True`, config loading, model initialization, checkpoint-loading strictness, device handling and eval mode are unchanged. Empty/invalid overrides are rejected rather than silently selecting another asset.

For existing helper-based entry points, set the environment variables in the command's environment. They do not globally override all trainers or fusion models: those that construct assets directly retain their own existing `--text_encoder` interfaces. Both paths must point to the same verified tokenizer/model assets used by the intended experiment; a different model identifier is not established as equivalent merely because it loads.

The originals are preserved separately in [legacy source snapshot](../archive/source_snapshots/evaluation/legacy_src.py) and [modern source snapshot](../archive/source_snapshots/evaluation/modern_src.py). [The change record](audit/portability_changes.csv) records original/current hashes; [the migration manifest](audit/applied_migration.csv) uses `original_snapshot_path` for these two intentional IO revisions. No other original function bodies or model/method sources are changed by this feature.

## Bind historical config paths into a new local copy

Use the existing Python/PyYAML environment. This is an **unexecuted user-data template**:

```bash
python -B tools/prepare_config.py \
  --source configs/historical/original_configs/Pretrain_anatomy_prior_A1.yaml \
  --output-dir configs/local/my-run \
  --set-path 'train_file=["/absolute/path/to/the-verified-manifest.json"]' \
  --set-path bert_config=/absolute/path/to/config_bert.json
```

The new directory receives `config.yaml` and `provenance.json`; the latter records source/output hashes, bindings and path-existence observations. Actual asset identity, split membership and run provenance are not established (`assets_verified` remains false).

Use `--paths-json /path/to/bindings.json` for a JSON object of dotted existing keys to absolute paths/path lists. Examples of supported existing fields include `bert_config`, `train_file`, `vindr_validation.labels_csv`, mask/image roots and `initial_checkpoints.original`. Duplicate assignments are rejected.

The tool rejects hyperparameter changes, unknown keys, activating null/empty fields, scalar/list changes, changed list cardinality and relative replacement paths. It preserves unrelated values, including fields that share YAML aliases. It refuses any existing output directory, including a dangling symlink. Inside the checkout, output is restricted to ignored `configs/local/`; other new directories must be outside the checkout.

This does not rewrite images, captions, manifests, checkpoints, notebook cells, prompts, augmentation or losses. Preserve authorized original assets and verify the manifest/weights themselves before execution. Historical configs remain immutable; local rebindings are not newly verified final thesis configs.

## Validation scope

The portability checks cover default/override precedence in both helpers, actual local synthetic-vocabulary tokenization and checkpoint IO with the unavailable pretrained model constructor isolated, AST equivalence of original functions outside the two routed asset arguments, and local-config safeguards. No clinical model inference or final checkpoint reproduction is claimed.

A separate CPU candidate environment was explored outside the repository before the priority shifted to model-code accessibility. No candidate lockfile or recovered HPC environment is published by this change. The existing legacy dependency declarations/environment remain unchanged; its newer Pillow-API mismatch remains documented.
