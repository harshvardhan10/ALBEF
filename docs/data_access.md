# Data access and prepared artifacts

Use authorized dataset access and local storage. No new dataset, image, report, checkpoint or model weight is added by this documentation phase. Existing tracked restricted-looking material has not been deleted or certified for redistribution.

## Dataset roles

| Dataset / artifact | Role | Verification needed |
|---|---|---|
| MIMIC-CXR image–text pairs | Training | Authorized source/version; report-to-caption processing; prepared manifest hash |
| CheXmask CSV/RLE data | Anatomical masks | Source release/license/hash; identifier alignment; quality/coverage criteria |
| Common-valid MIMIC manifest | Matched original/lung/heart population | Membership of the 43 exclusions; identical population across applicable runs |
| VinDr training-derived validation | Checkpoint selection, calibration, ensemble fitting | Frozen 2,000 IDs, labels, hashes and source split |
| VinDr official test | Final classification and localization | Frozen 3,000-image manifest, annotations and original dimensions |
| Positive-box cohorts | Pointing game | All eligible 309 Cardiomegaly and 111 PE pairs, complete coverage |

The checked-in training JSON has 215,394 entries with `image` and `caption`; the debug JSON has 200. Both use absolute image paths and contain captions. The thesis's prepared common-valid count is 215,351. Historical filtering explains the numerical difference but is not a substitute for the generated manifest.

## Preparation code

1. [Build CheXmask SQLite index](../scripts/build_chexmask_sqlite_index.py).
2. [Validate/filter manifests](../scripts/filter_chexmask_valid_manifest.py).
3. [Build mask cache](../scripts/precompute_chexmask_mask_cache.py) or preserve the distinct [precomputed-view path](../scripts/precompute_chexmask_common_views.py).
4. [Finalize mask manifests](../scripts/finalize_chexmask_mask_manifests.py) or [precomputed manifests](../scripts/finalize_chexmask_precomputed_manifests.py).
5. [Build fixed multi-label validation subset](../scripts/build_vindr_multilabel_val_subset.py).
6. [Build strict test-mask manifest](../scripts/build_vindr_test_mask_manifest.py).

These are separate commands with dataset-specific inputs; recover the original invocation and source checksums before claiming identical prepared data. Preserve deterministic sharding, original-resolution masks, view identity, quality thresholds and image geometry. Never substitute another mask release silently.

The [segmentation-derived heart-prior builder](../scripts/build_vindr_heart_prior_xrv.py) belongs to historical anatomical-prior work; its output is not the rectangular support specified for thesis A3.

## Local paths and publication

Future portable configuration should name separate data, tokenizer, checkpoint and output roots. Such overrides are proposed, not implemented. Historical configs retain their HPC paths for provenance. Put any local config copies outside the checkout until ignored paths are deliberately introduced.

Do not overwrite source images or reports during conversion. Preserve data licenses and access restrictions. Patient-image previews, contact sheets, report-bearing manifests, diagnostic IDs and notebook outputs require explicit review before public distribution. Do not add missing private data to Git to satisfy reproducibility documentation.
