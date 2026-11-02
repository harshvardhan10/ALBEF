"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../../archive/upstream_tasks/Pretrain_nlvr.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../../archive/upstream_tasks/Pretrain_nlvr.py').resolve()), 'exec'))
