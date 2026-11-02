#!/usr/bin/env python3
"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../tools/preprocessing/precompute_chexmask_common_views.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../tools/preprocessing/precompute_chexmask_common_views.py').resolve()), 'exec'))
