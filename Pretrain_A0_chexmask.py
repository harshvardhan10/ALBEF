"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / 'experiments/single_view/Pretrain_A0_chexmask.py').read_bytes(),
             str((_MigrationPath(__file__).parent / 'experiments/single_view/Pretrain_A0_chexmask.py').resolve()), 'exec'))
