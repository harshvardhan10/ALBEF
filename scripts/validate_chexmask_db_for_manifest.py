"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../tools/inspection/validate_chexmask_db_for_manifest.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../tools/inspection/validate_chexmask_db_for_manifest.py').resolve()), 'exec'))
