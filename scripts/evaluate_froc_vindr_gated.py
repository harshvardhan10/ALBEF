"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../archive/exploratory_localization/scripts/evaluate_froc_vindr_gated.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../archive/exploratory_localization/scripts/evaluate_froc_vindr_gated.py').resolve()), 'exec'))
