#!/usr/bin/env python3
"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../experiments/late_fusion/ensemble_zero_shot_scores.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../experiments/late_fusion/ensemble_zero_shot_scores.py').resolve()), 'exec'))
