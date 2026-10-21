# Data preparation

The original dataset manifests and configs remain unchanged as run evidence. Do not add patient data to Git. The imported preparation code is in cluster/data_preparation; historical preparation jobs and download helpers are under cluster/historical/scripts. Notebooks are under notebooks/data_preparation.

Preparation code requires authorized local datasets. No download/submission is executed during cleanup. Read each script's arguments and original launch record before use; cluster credentials and access are not configured here.

For the heatmap smoke test, use a local CSV with image_id first and binary Cardiomegaly and Pleural effusion columns. Freeze a small representative case selection before examining new maps. Images resolve as `<images_root>/<image_id>.png`; masks resolve as `<mask_root>/<first-two-ID-characters>/<image_id>.png`. Original and mask dimensions must agree. IDs must be unique simple filenames. Missing files fail rather than silently change the cohort.

The tracked data/mimic_cxr.json remains pending explicit local-preserving untracking approval. No history rewriting will be used to remove historical content.
