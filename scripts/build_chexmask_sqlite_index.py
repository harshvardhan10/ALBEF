"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../tools/preprocessing/build_chexmask_sqlite_index.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../tools/preprocessing/build_chexmask_sqlite_index.py').resolve()), 'exec'))
