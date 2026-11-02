"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../experiments/classification/bare_prompt/zero_shot_eval_vindr_pos_neg_bare_prompts.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../experiments/classification/bare_prompt/zero_shot_eval_vindr_pos_neg_bare_prompts.py').resolve()), 'exec'))
