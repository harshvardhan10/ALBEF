"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../tools/preprocessing/finalize_chexmask_mask_manifests.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../tools/preprocessing/finalize_chexmask_mask_manifests.py').resolve()), 'exec'))
