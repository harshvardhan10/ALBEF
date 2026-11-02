"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../experiments/external_baselines/biovil_t/extract_biovil_t_same_albef_cases.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../experiments/external_baselines/biovil_t/extract_biovil_t_same_albef_cases.py').resolve()), 'exec'))
