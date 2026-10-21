#!/usr/bin/env python3
"""Compare trusted, locally generated baseline/refactored heatmap outputs."""
import argparse
import json
import math
from pathlib import Path

import torch


def compare_payloads(baseline, candidate, rtol=1e-5, atol=1e-6):
    failures, differences = [], {}

    def validate(data, name):
        if not isinstance(data, dict) or not isinstance(data.get('__metadata__'), dict):
            failures.append(name + ': missing metadata')
            return
        labels = [label for label in data if label != '__metadata__']
        if not labels:
            failures.append(name + ': no label results')
        for label in labels:
            result = data[label]
            if not isinstance(result, dict):
                failures.append(name + ': invalid label result ' + label)
                continue
            for key in ['positive_prompt', 'negative_prompt', 'ground_truth', 'positive_similarity',
                        'negative_similarity', 'margin', 'classification_logit', 'positive_probability']:
                if key not in result:
                    failures.append(name + '/' + label + ': missing ' + key)
            for branch in ['original', 'lung', 'heart', 'fused']:
                cam = result.get(branch, {})
                for field in ['cam_signed_raw', 'cam_positive_raw', 'cam_vis', 'cam_vis_up']:
                    value = cam.get(field) if isinstance(cam, dict) else None
                    if not torch.is_tensor(value) or value.ndim != 2 or not value.numel():
                        failures.append(name + '/' + label + '/' + branch + ': invalid ' + field)

    validate(baseline, 'baseline')
    validate(candidate, 'candidate')
    if failures:
        return {'passed': False, 'failures': failures, 'absolute_differences': {}, 'max_absolute_difference': 0.}

    def compare(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                failures.append(path + ': keys differ')
            for key in sorted(set(a) & set(b)):
                compare(a[key], b[key], path + '/' + key)
        elif torch.is_tensor(a) and torch.is_tensor(b):
            if a.shape != b.shape:
                failures.append(path + ': shape differs')
            elif not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                failures.append(path + ': nonfinite tensor')
            else:
                differences[path] = float((a.float() - b.float()).abs().max()) if a.numel() else 0.
                if not torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol):
                    failures.append(path + ': tensor differs')
        elif isinstance(a, (float, int)) and isinstance(b, (float, int)):
            if not (math.isfinite(a) and math.isfinite(b)):
                failures.append(path + ': nonfinite scalar')
            else:
                differences[path] = abs(a - b)
                if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
                    failures.append(path + ': scalar differs')
        elif type(a) is not type(b) or a != b:
            failures.append(path + ': value differs')

    # Paths and source revision can change; scientific identity must not.
    ignored = {'checkpoint', 'config', 'image_path', 'lung_mask_path', 'heart_mask_path'}
    a = {**baseline, '__metadata__': {k: v for k, v in baseline.get('__metadata__', {}).items() if k not in ignored}}
    b = {**candidate, '__metadata__': {k: v for k, v in candidate.get('__metadata__', {}).items() if k not in ignored}}
    compare(a, b, 'payload')
    return {'passed': not failures, 'failures': failures, 'absolute_differences': differences,
            'max_absolute_difference': max(differences.values(), default=0.)}


def compare_directories(baseline, candidate, rtol=1e-5, atol=1e-6):
    if rtol < 0 or atol < 0 or not math.isfinite(rtol + atol):
        raise ValueError('Tolerances must be finite and nonnegative')
    filename = 'cross_attention_fusion_itc_margin_gradcam_manifest.json'
    a = json.loads((baseline/filename).read_text())
    b = json.loads((candidate/filename).read_text())
    invariant_keys = ['checkpoint_sha256', 'checkpoint_role', 'training_commit', 'config_sha256', 'resolved_config',
                      'labels_csv_sha256', 'case_ids', 'method', 'fusion_type', 'target_labels', 'prompts',
                      'temperature', 'vit_layer_index', 'image_res', 'num_vit_layers', 'normalization',
                      'interpolation', 'anchor_view', 'memory_views', 'max_text_length', 'environment', 'num_images']
    failures = []
    for key in invariant_keys:
        if key not in a or key not in b or a[key] != b[key]:
            failures.append('manifest mismatch/missing: ' + key)
    expected = {str(x) + '.pt' for x in a.get('case_ids', [])}
    if not expected or len(expected) != a.get('num_images'):
        failures.append('baseline manifest case count invalid')
    actual_a = {x.name for x in baseline.glob('*.pt')}
    actual_b = {x.name for x in candidate.glob('*.pt')}
    if actual_a != expected or actual_b != expected:
        failures.append('case files differ from expected manifest IDs')
    cases = {}
    for name in sorted(expected & actual_a & actual_b):
        cases[name] = compare_payloads(torch.load(baseline/name, map_location='cpu'),
                                      torch.load(candidate/name, map_location='cpu'), rtol, atol)
        if not cases[name]['passed']:
            failures.append('case differs: ' + name)
    return {'passed': not failures, 'rtol': rtol, 'atol': atol, 'failures': failures, 'cases': cases,
            'baseline_commit': a.get('git_commit'), 'candidate_commit': b.get('git_commit')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-6)
    args = parser.parse_args()
    result = compare_directories(args.baseline, args.candidate, args.rtol, args.atol)
    with args.report.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print('PASS' if result['passed'] else 'FAIL', args.report)
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
