# Data preparation

The original dataset manifests and configs remain unchanged as run evidence. Do not add patient data to Git. The imported preparation code is in cluster/data_preparation; historical preparation jobs and download helpers are under cluster/historical/scripts. Notebooks are under notebooks/data_preparation.

Preparation code requires authorized local datasets. No download/submission is executed during cleanup. Read each script's arguments and original launch record before use; cluster credentials and access are not configured here.

For the heatmap smoke test, use a local CSV with image_id first and binary Cardiomegaly and Pleural effusion columns. Freeze a small representative case selection before examining new maps. Images resolve as `<images_root>/<image_id>.png`; masks resolve as `<mask_root>/<first-two-ID-characters>/<image_id>.png`. Original and mask dimensions must agree. IDs must be unique simple filenames. Missing files fail rather than silently change the cohort.

data/mimic_cxr.json is now ignored and untracked with user approval; its local working copy is preserved. On an older server checkout, follow the backup step in cross_attention_refactor_check.md before pulling this removal. No Git history is rewritten. A synthetic schema example is provided in data/mimic_cxr.example.json; it contains no usable patient data. The pretraining loader requires image and caption fields and accepts either one caption or a list of captions. Keep the archived historical manifest for exact reproduction.

## Regenerating a manifest when needed

The imported preparation tool creates frontal image/text pairs from authorized MIMIC-CXR-JPG metadata and NegBio labels. This is an existing preparation route, not proof of reproducing the exact historical manifest. Inspect its options first:

```bash
python cluster/data_preparation/prepare_mimic_albef.py --help
```

Use `--base-dir` for extracted image files, `--negbio-csv` and `--metadata-csv` for local source tables, and `--output-json` for a new local manifest file. For a small preparation check, use `--max-studies 2`. Omit `--relative-to` for root pretraining loaders that expect directly readable image paths. The tool can write an existing output path, so choose a fresh filename and keep the archived manifest unchanged. No preparation job has been run during cleanup.
