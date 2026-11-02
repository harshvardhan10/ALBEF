"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../archive/non_thesis_experiments/anatomy_prior/patch_head.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../archive/non_thesis_experiments/anatomy_prior/patch_head.py').resolve()), 'exec'))
