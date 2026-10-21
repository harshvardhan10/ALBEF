"""Construct fusion modules without changing historical parameter names."""
from .fc import FullyConnectedFusion
from .mean import SimpleMeanFusion
from .gated import ImageLevelGatedFusion
from .residual import OriginalAnchoredResidualFusion
from .transformer import ViewTransformerFusion
from .cross_attention import ViewCrossAttentionFusion

FUSION_NAMES = ("fc", "mean", "gated", "residual", "transformer", "cross_attention")


def build_fusion_module(name, hidden_dim, config):
    if name == "fc":
        return FullyConnectedFusion(hidden_dim)
    if name == "mean":
        return SimpleMeanFusion()
    if name == "gated":
        cfg = config.get("view_fusion_gate", {})
        return ImageLevelGatedFusion(hidden_dim=hidden_dim, gate_hidden_dim=int(cfg.get("hidden_dim", hidden_dim)))
    if name == "residual":
        cfg = config.get("view_fusion_residual", {})
        return OriginalAnchoredResidualFusion(hidden_dim=hidden_dim, gate_hidden_dim=int(cfg.get("hidden_dim", hidden_dim)), initial_gate=float(cfg.get("initial_gate", 0.1)))
    if name in ("transformer", "cross_attention"):
        key = "view_fusion_transformer" if name == "transformer" else "view_fusion_cross_attention"
        cfg = config.get(key, {})
        if name == "cross_attention":
            if str(cfg.get("anchor_view", "original")).lower() != "original":
                raise ValueError("Anchored cross-attention ablation requires anchor_view=original")
            if [str(v).lower() for v in cfg.get("memory_views", ["lung", "heart"])] != ["lung", "heart"]:
                raise ValueError("Anchored cross-attention ablation requires memory_views=[lung, heart]")
        module = ViewTransformerFusion if name == "transformer" else ViewCrossAttentionFusion
        return module(hidden_dim=hidden_dim, num_heads=int(cfg.get("num_heads", 12)),
                      num_layers=int(cfg.get("num_layers", 2)), mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
                      dropout=float(cfg.get("dropout", 0.0)))
    raise ValueError("Unknown fusion type: " + str(name))
