"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../archive/exploratory_localization/scripts/evaluate_heatmaps_vindr.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../archive/exploratory_localization/scripts/evaluate_heatmaps_vindr.py').resolve()), 'exec'))
