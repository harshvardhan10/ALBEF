"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../experiments/localization/historical/extract_albef_cross_attn_gradcam.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../experiments/localization/historical/extract_albef_cross_attn_gradcam.py').resolve()), 'exec'))
