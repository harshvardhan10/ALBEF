"""Historical import path; implementation lives in the shared multiview model."""
from models.model_pretrain_multiview import ALBEF as SharedALBEF, concat_all_gather
from models.fusion.mean import SimpleMeanFusion


class ALBEF(SharedALBEF):
    DEFAULT_FUSION = "mean"
