"""Provenance for new artifacts; missing historical evidence stays unknown."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

from evaluation.prompts import build_prompts_for_label
from evaluation.checkpoint_roles import CHECKPOINT_ROLES

ROOT = Path(__file__).resolve().parents[1]


def repository_state():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = 'unavailable', None
    return {'git_commit': commit, 'git_dirty': dirty, 'python': platform.python_version()}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_score_metadata(output_path, outputs, metrics, context=None):
    labels = list(map(str, outputs['label_names']))
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Score columns need nonempty unique labels')
    role = (context or {}).get('checkpoint_role')
    if role is not None and role not in CHECKPOINT_ROLES:
        raise ValueError('Unknown checkpoint role: ' + str(role))
    record = {
        'schema_version': 1, **repository_state(), 'artifact_kind': 'validation_epoch_scores',
        'artifact_sha256': sha256_file(output_path), 'label_order': labels,
        'num_images': len(outputs['image_ids']), 'epoch': int(metrics.epoch),
        'view_type': metrics.view_type, 'method': 'ITC cosine similarity with averaged prompt embeddings',
        'prompts': {label: build_prompts_for_label(label) for label in labels},
        'checkpoint_role': role,
        'threshold_source': 'current validation split; not a held-out test threshold',
        'run_context': context,
    }
    path = Path(str(output_path)+'.metadata.json')
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)
