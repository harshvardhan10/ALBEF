"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../archive/exploratory_localization/scripts/compute_froc_from_gradcam_heatmaps_revised.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../archive/exploratory_localization/scripts/compute_froc_from_gradcam_heatmaps_revised.py').resolve()), 'exec'))
