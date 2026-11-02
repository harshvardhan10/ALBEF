"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../tools/inspection/inspect_chexmask_excluded.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../tools/inspection/inspect_chexmask_excluded.py').resolve()), 'exec'))
