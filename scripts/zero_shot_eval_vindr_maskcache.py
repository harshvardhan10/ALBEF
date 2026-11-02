"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../experiments/classification/historical/zero_shot_eval_vindr_maskcache.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../experiments/classification/historical/zero_shot_eval_vindr_maskcache.py').resolve()), 'exec'))
