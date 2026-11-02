"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../src/cxr_vlm/evaluation/localization/vindr_online_validation.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../src/cxr_vlm/evaluation/localization/vindr_online_validation.py').resolve()), 'exec'))
