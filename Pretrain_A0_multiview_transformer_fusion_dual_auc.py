"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / 'experiments/joint_fusion/transformer/Pretrain_A0_multiview_transformer_fusion_dual_auc.py').read_bytes(),
             str((_MigrationPath(__file__).parent / 'experiments/joint_fusion/transformer/Pretrain_A0_multiview_transformer_fusion_dual_auc.py').resolve()), 'exec'))
