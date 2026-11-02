"""Compatibility checks for structural migration; no private data or downloads."""
import ast
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class MigrationTests(unittest.TestCase):
    def run_python(self, *args, cwd=ROOT):
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONPATH'] = str(ROOT) + os.pathsep + str(ROOT / 'src')
        return subprocess.run([sys.executable, '-B', *map(str, args)], cwd=cwd,
                              env=env, capture_output=True, text=True, timeout=60)

    def test_original_migrated_contents_are_exact(self):
        manifest = ROOT / 'docs/audit/applied_migration.csv'
        self.assertTrue(manifest.is_file(), 'Applied migration manifest is missing')
        with manifest.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 175)
        self.assertEqual(len(rows), len({r['canonical_path'] for r in rows}))
        for row in rows:
            with self.subTest(path=row['canonical_path']):
                source = ROOT / row['canonical_path']
                self.assertTrue(source.is_file())
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                                 row['original_sha256'])
                legacy = ROOT / row['legacy_path']
                self.assertTrue(legacy.is_file())
                if row['kind'] == 'config':
                    self.assertTrue(legacy.is_symlink())
                    self.assertEqual(legacy.resolve(), source.resolve())
                else:
                    self.assertFalse(legacy.is_symlink())
                    tree = ast.parse(legacy.read_text())
                    targets = []
                    for call in ast.walk(tree):
                        if (isinstance(call, ast.Call) and
                            isinstance(call.func, ast.Attribute) and
                            call.func.attr == 'read_bytes'):
                            operand = call.func.value
                            self.assertIsInstance(operand, ast.BinOp)
                            self.assertIsInstance(operand.op, ast.Div)
                            relative = ast.literal_eval(operand.right)
                            targets.append((legacy.parent / relative).resolve())
                    self.assertEqual(targets, [source.resolve()])

    def test_legacy_and_canonical_launcher_help_match(self):
        launcher = ROOT / 'tools/run_experiment.py'
        self.assertTrue(launcher.is_file(), 'Canonical launcher is missing')
        for old, new in [
            ('Pretrain.py', 'experiments/baseline/Pretrain.py'),
            ('Pretrain_A0_multiview_cross_attention_fusion_dual_auc.py',
             'experiments/joint_fusion/cross_attention/Pretrain_A0_multiview_cross_attention_fusion_dual_auc.py'),
            ('scripts_new/extract_itc_margin_gradcam_heatmaps.py',
             'experiments/localization/itc/extract_itc_margin_gradcam_heatmaps.py'),
            ('scripts/build_vindr_checkpoint_val_split.py',
             'tools/preprocessing/build_vindr_checkpoint_val_split.py')]:
            with self.subTest(path=old):
                legacy = self.run_python(ROOT / old, '--help')
                current = self.run_python(launcher, new, '--help')
                self.assertEqual(legacy.returncode, 0, legacy.stderr)
                self.assertEqual(current.returncode, 0, current.stderr)
                self.assertEqual(current.stdout, legacy.stdout)

    def test_launcher_rejects_unregistered_paths(self):
        launcher = ROOT / 'tools/run_experiment.py'
        self.assertTrue(launcher.is_file(), 'Canonical launcher is missing')
        result = self.run_python(launcher, '../unregistered.py')
        self.assertEqual(result.returncode, 2)
        self.assertIn('registered', result.stderr)

    def test_launcher_requires_repository_working_directory(self):
        launcher = ROOT / 'tools/run_experiment.py'
        self.assertTrue(launcher.is_file(), 'Canonical launcher is missing')
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_python(launcher, 'Pretrain.py', '--help', cwd=temporary)
        self.assertEqual(result.returncode, 2)
        self.assertIn('repository root', result.stderr)

    def test_legacy_helpers_resolve_with_src_layout(self):
        self.assertTrue((ROOT / 'src').is_dir(), 'src layout is missing')
        source = importlib.import_module('src')
        historical = importlib.import_module('scripts.src')
        self.assertIs(source.get_label_text_embeddings,
                      historical.get_label_text_embeddings)

    def test_canonical_reusable_imports_and_state_keys(self):
        self.assertTrue((ROOT / 'src/cxr_vlm').is_dir(), 'Canonical package is missing')
        sys.path.insert(0, str(ROOT / 'src'))
        import torch
        old = importlib.import_module('models.vit')
        new = importlib.import_module('cxr_vlm.models.vit')
        torch.manual_seed(7)
        a = old.VisionTransformer(img_size=32, patch_size=16, embed_dim=32,
                                  depth=2, num_heads=4)
        torch.manual_seed(7)
        b = new.VisionTransformer(img_size=32, patch_size=16, embed_dim=32,
                                  depth=2, num_heads=4)
        image = torch.arange(3072, dtype=torch.float32).reshape(1, 3, 32, 32) / 3072
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))
        torch.testing.assert_allclose(a(image), b(image), rtol=0, atol=0)
        validation = importlib.import_module(
            'cxr_vlm.evaluation.classification.vindr_multiview_classification_validation')
        self.assertTrue(callable(validation.build_validation_transform))

    def test_historical_executable_entrypoints(self):
        paths = [
            'scripts/ensemble_crossattn_gradcam_heatmaps.py',
            'scripts/ensemble_zero_shot_scores.py',
            'scripts/finalize_chexmask_precomputed_manifests.py',
            'scripts/precompute_chexmask_common_views.py',
        ]
        for relative in paths:
            with self.subTest(path=relative):
                path = ROOT / relative
                self.assertTrue(os.access(path, os.X_OK), 'Historical executable mode was lost')
                self.assertTrue(path.read_text().startswith('#!'), 'Historical shebang was lost')
        env = os.environ.copy()
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env['PATH']
        env['PYTHONPATH'] = str(ROOT)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        result = subprocess.run([str(ROOT / paths[1]), '--help'], cwd=ROOT,
                                env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('usage:', result.stdout)

    def test_preview_root_remains_historical(self):
        preview = importlib.import_module('scripts.preview_chexmask_views')
        self.assertEqual(preview.PROJECT_ROOT, ROOT)

    def test_fixed_image_model_and_fusion_behavior(self):
        from behavior_probe import capture, summarize
        expected = json.loads((Path(__file__).parent / 'reference.json').read_text())
        actual = summarize(capture())
        def compare(want, got, path='root'):
            if isinstance(want, dict):
                self.assertEqual(set(want), set(got), path)
                for key in want:
                    compare(want[key], got[key], path + '.' + key)
            elif isinstance(want, list):
                self.assertEqual(len(want), len(got), path)
                for i, (a, b) in enumerate(zip(want, got)):
                    compare(a, b, path + '[' + str(i) + ']')
            elif isinstance(want, float):
                self.assertAlmostEqual(want, got, delta=1e-6 * max(1., abs(want)), msg=path)
            else:
                self.assertEqual(want, got, path)
        compare(expected, actual)


if __name__ == '__main__':
    unittest.main()
