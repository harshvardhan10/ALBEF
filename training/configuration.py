"""Resolve portable experiment configs without editing historical YAML files."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import yaml


def _read(path):
    with Path(path).open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Expected YAML mapping: " + str(path))
    return value


def _path_key(key):
    return key.endswith(("_file", "_files", "_root", "_path", "_csv", "_dir", "_checkpoint")) or key in ("bert_config", "text_encoder", "checkpoint", "pretrained")


def load_run_config(experiment_path, environment_path=None):
    result = copy.deepcopy(_read(experiment_path))
    if environment_path is None:
        return result
    environment = _read(environment_path)
    for key, value in environment.items():
        if key in ("initial_checkpoints", "vindr_validation"):
            if not isinstance(value, dict):
                raise ValueError("Environment section must be a mapping: " + key)
            allowed = {"original", "lung", "heart"} if key == "initial_checkpoints" else {"labels_csv", "images_root", "lung_mask_root", "heart_mask_root", "mask_root"}
            if set(value) - allowed:
                raise ValueError("Environment cannot change methodology in " + key)
            result.setdefault(key, {}).update(value)
        elif _path_key(key):
            result[key] = value
        else:
            raise ValueError("Environment cannot override methodological field: " + key)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_run_config(args.experiment, args.environment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(args.output)


if __name__ == "__main__":
    main()
