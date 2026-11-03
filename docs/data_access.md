# Data access and prepared artifacts

## Preparation code

1. [Build CheXmask SQLite index](../tools/preprocessing/build_chexmask_sqlite_index.py).
2. [Validate/filter manifests](../tools/preprocessing/filter_chexmask_valid_manifest.py).
3. [Build mask cache](../tools/preprocessing/precompute_chexmask_mask_cache.py) or preserve the distinct [precomputed-view path](../tools/preprocessing/precompute_chexmask_common_views.py).
4. [Finalize mask manifests](../tools/preprocessing/finalize_chexmask_mask_manifests.py) or [precomputed manifests](../tools/preprocessing/finalize_chexmask_precomputed_manifests.py).
5. [Build fixed multi-label validation subset](../tools/preprocessing/build_vindr_multilabel_val_subset.py).
6. [Build strict test-mask manifest](../tools/preprocessing/build_vindr_test_mask_manifest.py).
