"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../tools/preprocessing/build_vindr_heart_prior_xrv.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../tools/preprocessing/build_vindr_heart_prior_xrv.py').resolve()), 'exec'))
