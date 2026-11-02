"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../experiments/late_fusion/learn_dual_checkpoint_ensemble_weights.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../experiments/late_fusion/learn_dual_checkpoint_ensemble_weights.py').resolve()), 'exec'))
