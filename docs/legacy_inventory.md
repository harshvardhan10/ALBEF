# Compatibility and historical inventory

| Material | Disposition and reason |
|---|---|
| Six old multiview model modules | Compatibility subclasses; implementation in models/model_pretrain_multiview.py and models/fusion/ |
| Six old multiview training commands | Compatibility CLI/helper exports; implementation in training/ |
| Anatomy/A4/A5/A6.1 trainers | Separate training loops retained; shared support/checkpoint/patch helpers extracted |
| scripts/src.py and scripts_new/src.py | Compatibility exports of the originally identical ALBEF utility implementation in evaluation/albef_utils.py |
| Four scripts/*FROC* implementations | Exploratory source under scripts/exploratory_froc/, original command/import paths retained as wrappers; relocation.json records hashes |
| cluster/historical/ | Unchanged launch provenance, including machine paths and historical mail directives; do not submit unchanged |
| Original configs/ and configs_new/ YAML | Historical evidence unchanged; portable presets live separately |
| Original ALBEF VQA/NLVR/retrieval/single-view code | Retained upstream/historical implementation and import support; absence from the thesis reading path does not prove safe deletion |
| Original ALBEF notebooks/results | Retained scientific/historical evidence; indexed rather than automatically stripped or deleted |
| tests/references/*.json.gz | Frozen original model/trainer source used independently by equivalence tests |
| requirements/notebooks-historical.txt | Exact original root dependency freeze, not the successful training environment |
| 70 generated .pyc plus data/mimic_cxr.json | Explicitly approved untracking; local copies preserved, ignored; no Git history rewrite |

The original full-file inventory is docs/cleanup_inventory.json. Companion inclusions/exclusions and modified notebook hashes are in docs/source_imports.json. No additional deletion is inferred from this classification.
