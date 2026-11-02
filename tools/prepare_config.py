#!/usr/bin/env python3
"""Create a new local path-bound config; never edit historical configs or results."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PATH_FIELDS = frozenset({
    'train_file', 'val_file', 'test_file', 'bert_config', 'labels_csv',
    'image_root', 'images_root', 'mask_root', 'lung_train_file', 'heart_train_file',
    'lung_mask_root', 'heart_mask_root', 'chexmask_db', 'anatomy_prior_path',
    'classification_teacher_checkpoint', 'vqa_root', 'vg_root', 'refcoco_data',
    'dets_file', 'coco_file', 'meta_csv', 'metadata_csv', 'index_csv',
})


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate path assignment: ' + key)
        result[key] = value
    return result


def load_overrides(paths_json, assignments):
    """Read dotted path bindings and reject duplicates instead of silently winning."""
    result = {}
    if paths_json is not None:
        result = json.loads(Path(paths_json).read_text(), object_pairs_hook=_unique_object)
        if not isinstance(result, dict):
            raise ValueError('Path JSON must be an object of dotted keys to paths')
    for assignment in assignments:
        key, separator, value = assignment.partition('=')
        if not separator or not key:
            raise ValueError('Use dotted.key=/absolute/path or a JSON path list')
        if key in result:
            raise ValueError('Duplicate path assignment: ' + key)
        result[key] = json.loads(value) if value.startswith('[') else value
    return result


def _existing_value(config, parts):
    value = config
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            raise ValueError('Path field does not exist: ' + '.'.join(parts))
        value = value[part]
    return value


def _absolute_path(value):
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError('Replacement paths must be nonempty absolute strings')
    return value


def _bind(config, parts, value):
    # Functional branch copies avoid mutating another field through a YAML alias.
    bound = dict(config)
    if len(parts) == 1:
        bound[parts[0]] = value
    else:
        bound[parts[0]] = _bind(config[parts[0]], parts[1:], value)
    return bound


def prepare_config(source, output_dir, overrides):
    """Write config.yaml/provenance.json into a new private/local directory.

    Only existing allowlisted path fields may change. Disabled/null fields,
    scalar/list shape changes and list cardinality changes are rejected.
    Asset existence is recorded, not treated as proof of dataset/model identity.
    """
    source = Path(source).resolve()
    requested = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(requested):
        raise FileExistsError('Output directory already exists: ' + str(requested))
    output_dir = requested.resolve()
    local = ROOT / 'configs/local'
    for location in (requested, output_dir):
        if ROOT == location or ROOT in location.parents:
            if location != local and local not in location.parents:
                raise ValueError('Inside the checkout, use ignored configs/local/ only')
    raw = source.read_bytes()
    original = yaml.safe_load(raw)
    if not isinstance(original, dict) or not isinstance(overrides, dict) or not overrides:
        raise ValueError('A mapping config and at least one explicit path binding are required')
    bound = copy.deepcopy(original)
    existence = {}
    for key, value in overrides.items():
        if not isinstance(key, str) or not key or any(not p for p in key.split('.')):
            raise ValueError('Invalid dotted path field')
        parts = key.split('.')
        checkpoint = (len(parts) == 2 and parts[0] == 'initial_checkpoints'
                      and parts[1] in {'original', 'lung', 'heart'})
        if parts[-1] not in PATH_FIELDS and not checkpoint:
            raise ValueError('Not an approved path field: ' + key)
        previous = _existing_value(original, parts)
        if isinstance(previous, str) and previous:
            replacement = _absolute_path(value)
            existence[key] = Path(replacement).exists()
        elif (isinstance(previous, list) and previous and
              all(isinstance(v, str) and v for v in previous)):
            if not isinstance(value, list) or len(value) != len(previous):
                raise ValueError('Path list shape/cardinality must remain unchanged: ' + key)
            replacement = [_absolute_path(v) for v in value]
            existence[key] = [Path(v).exists() for v in replacement]
        else:
            raise ValueError('Cannot activate or change an inactive/non-path field: ' + key)
        bound = _bind(bound, parts, replacement)
    payload = yaml.safe_dump(bound, sort_keys=False).encode('utf-8')
    record = {
        'generated_by': 'tools/prepare_config.py',
        'source_config': str(source),
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'config_sha256': hashlib.sha256(payload).hexdigest(),
        'bindings': overrides,
        'path_exists': existence,
        'assets_verified': False,
        'note': 'Path existence does not verify asset identity, split membership or run provenance.',
    }
    provenance = json.dumps(record, indent=2) + '\n'
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / 'config.yaml'
    provenance_path = output_dir / 'provenance.json'
    with config_path.open('xb') as handle:
        handle.write(payload)
    with provenance_path.open('x') as handle:
        handle.write(provenance)
    return config_path, provenance_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output-dir', required=True, help='New local directory; never overwritten')
    parser.add_argument('--paths-json', default=None)
    parser.add_argument('--set-path', action='append', default=[], metavar='DOTTED.KEY=PATH')
    args = parser.parse_args()
    try:
        overrides = load_overrides(args.paths_json, args.set_path)
        config, provenance = prepare_config(args.source, args.output_dir, overrides)
    except (ValueError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(json.dumps({'config': str(config), 'provenance': str(provenance),
                      'assets_verified': False}, indent=2))


if __name__ == '__main__':
    main()
