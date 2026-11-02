"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../src/cxr_vlm/models/tokenization_bert.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../src/cxr_vlm/models/tokenization_bert.py').resolve()), 'exec'))
