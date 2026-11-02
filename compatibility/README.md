# Historical command compatibility

The 23 former root command wrappers are grouped in [entrypoints/](entrypoints/). The full research implementations remain in `experiments/`, `src/cxr_vlm/`, `tools/` and `archive/`.

Run registered commands from the repository root using the launcher:

```bash
python -B tools/run_experiment.py experiments/baseline/Pretrain.py --help
python -B tools/run_experiment.py Pretrain.py --help
```

Both names select the same implementation. Replace old commands such as `python Pretrain.py ...` with `python tools/run_experiment.py Pretrain.py ...`, retaining all experiment flags. The launcher restores the historical import search directories; it does not change the working directory, configuration values or data arguments. Root entrypoint `__file__` locations now refer to these grouped wrappers; the corresponding implementations do not use `__file__`.

The [applied migration manifest](../docs/audit/applied_migration.csv) maps `legacy_path` (accepted command alias), `canonical_path` (full implementation) and `compatibility_path` (physical wrapper). Imported helper wrappers remain at their historical paths for notebook imports, resource lookup and checkpoint module identities. Do not treat them as duplicate model implementations.
