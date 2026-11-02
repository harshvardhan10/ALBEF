"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../experiments/localization/itc/extract_anatomy_itc_margin_gradcam_a1_a5.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../experiments/localization/itc/extract_anatomy_itc_margin_gradcam_a1_a5.py').resolve()), 'exec'))
