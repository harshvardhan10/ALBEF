"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / 'archive/non_thesis_experiments/extract_a6_0_identity_patch_head_heatmaps.py').read_bytes(),
             str((_MigrationPath(__file__).parent / 'archive/non_thesis_experiments/extract_a6_0_identity_patch_head_heatmaps.py').resolve()), 'exec'))
