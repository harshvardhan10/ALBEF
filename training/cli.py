"""Backward-compatible command line for the six thesis fusion trainers."""
from __future__ import annotations
import argparse
import copy
import importlib
import sys
from pathlib import Path
import yaml
from training.configuration import load_run_config

FUSIONS = ("fc", "mean", "gated", "residual", "transformer", "cross_attention")

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower().strip()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def make_parser(fusion_type=None):
    parser = argparse.ArgumentParser(description="Train the thesis multiview ALBEF model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment_config", default=None)
    parser.add_argument("--fusion_type", choices=FUSIONS, default=fusion_type)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--auto_resume", type=str2bool, default=True)
    parser.add_argument("--original_checkpoint", default="")
    parser.add_argument("--lung_checkpoint", default="")
    parser.add_argument("--heart_checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--distributed", type=str2bool, default=True)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return parser


def run_legacy_main(fusion_type, args, config):
    config = copy.deepcopy(config)
    config["view_fusion_type"] = fusion_type
    from training.multiview_trainer import main
    return main(args, config)


def main_for_fusion(fusion_type=None, argv=None):
    parser = make_parser(fusion_type)
    args = parser.parse_args(argv)
    config = load_run_config(args.config, args.environment_config)
    chosen = args.fusion_type or config.get("view_fusion_type")
    if chosen not in FUSIONS or (fusion_type and chosen != fusion_type):
        parser.error("Choose a fusion consistent with the historical trainer")
    config["view_fusion_type"] = chosen
    explicit = list(argv) if argv is not None else sys.argv[1:]
    if not any(x == "--text_encoder" or x.startswith("--text_encoder=") for x in explicit):
        args.text_encoder = config.get("text_encoder", args.text_encoder)
    config["text_encoder"] = args.text_encoder
    for view in ("original", "lung", "heart"):
        override = getattr(args, view + "_checkpoint")
        if override:
            config.setdefault("initial_checkpoints", {})[view] = override
    from evaluation.experiment_metadata import repository_state, sha256_file
    config["_run_metadata"] = {**repository_state(), "source_config_sha256": sha256_file(args.config),
                               "environment_config": args.environment_config, "command_arguments": explicit}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/"config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    from training.multiview_trainer import main
    main(args, config)


def legacy_export(name, fusion_type):
    # Keep imports from old trainer modules usable without eager heavy imports.
    import inspect
    if name in ("SynchronizedCXRTransform", "MultiViewCXRPretrainDataset", "_as_path_list", "load_json_records", "_safe_relative_path", "ALBEF_MEAN", "ALBEF_STD"):
        return getattr(importlib.import_module("dataset.multiview_cxr_dataset"), name)
    if name == "ALBEF":
        suffix = "fusion" if fusion_type == "fc" else fusion_type + "_fusion"
        return importlib.import_module("models.model_pretrain_multiview_" + suffix).ALBEF
    value = getattr(importlib.import_module("training.multiview_trainer"), name)
    if inspect.isfunction(value) and "config" in inspect.signature(value).parameters:
        def selected(*args, **kwargs):
            bound = inspect.signature(value).bind(*args, **kwargs)
            config = copy.deepcopy(bound.arguments["config"])
            config["view_fusion_type"] = fusion_type
            bound.arguments["config"] = config
            return value(*bound.args, **bound.kwargs)
        return selected
    return value
