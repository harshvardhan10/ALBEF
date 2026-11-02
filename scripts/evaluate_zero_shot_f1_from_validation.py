"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../experiments/evaluation/evaluate_zero_shot_f1_from_validation.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../experiments/evaluation/evaluate_zero_shot_f1_from_validation.py').resolve()), 'exec'))
