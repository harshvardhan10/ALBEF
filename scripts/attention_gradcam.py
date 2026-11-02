"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../src/cxr_vlm/localization/legacy/attention_gradcam.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../src/cxr_vlm/localization/legacy/attention_gradcam.py').resolve()), 'exec'))
