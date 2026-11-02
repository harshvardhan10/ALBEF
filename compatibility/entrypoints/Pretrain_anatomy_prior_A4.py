"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../../experiments/anatomy_guidance/Pretrain_anatomy_prior_A4.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../../experiments/anatomy_guidance/Pretrain_anatomy_prior_A4.py').resolve()), 'exec'))
