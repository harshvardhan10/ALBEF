"""Historical compatibility path; implementation moved without behavior edits."""
from pathlib import Path as _MigrationPath
exec(compile((_MigrationPath(__file__).parent / '../experiments/joint_fusion/residual/localization/extract_residual_fusion_itc_margin_gradcam_heatmaps.py').read_bytes(),
             str((_MigrationPath(__file__).parent / '../experiments/joint_fusion/residual/localization/extract_residual_fusion_itc_margin_gradcam_heatmaps.py').resolve()), 'exec'))
