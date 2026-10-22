import ast
import copy
import gzip
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import utils
from training.configuration import load_run_config
from training.cli import make_parser
from training.multiview_trainer import train_one_epoch, cleanup_last_checkpoint, experiment_metadata
from training.multiview_checkpoints import resume_checkpoint
from evaluation.checkpoint_roles import DualAUCSelectionState
from tests.multiview_support import ROOT, VARIANTS

with gzip.open(ROOT/'tests/references/training_6320a2e.json.gz', 'rt') as handle:
    REFERENCES = json.load(handle)


class ConfigurationTests(unittest.TestCase):
    def test_single_view_mask_and_classification_teacher_paths_can_be_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'experiment.yaml').write_text('classification_teacher_checkpoint: null\nvindr_validation:\n  mask_root: null\n')
            (root/'environment.yaml').write_text('classification_teacher_checkpoint: /weights/teacher.pth\nvindr_validation:\n  mask_root: /masks/lung\n')
            result = load_run_config(root/'experiment.yaml', root/'environment.yaml')
            self.assertEqual(result['classification_teacher_checkpoint'], '/weights/teacher.pth')
            self.assertEqual(result['vindr_validation']['mask_root'], '/masks/lung')

    def test_environment_overrides_paths_but_cannot_change_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'experiment.yaml').write_text('view_fusion_type: mean\nimage_res: 256\nlung_mask_root: null\nvindr_validation:\n  labels_csv: null\n  enabled: true\n')
            (root/'environment.yaml').write_text('lung_mask_root: /data/lung\nvindr_validation:\n  labels_csv: /data/labels.csv\n')
            config = load_run_config(root/'experiment.yaml', root/'environment.yaml')
            self.assertEqual(config['lung_mask_root'], '/data/lung')
            self.assertEqual(config['vindr_validation'], {'labels_csv': '/data/labels.csv', 'enabled': True})
            (root/'environment.yaml').write_text('image_res: 384\n')
            with self.assertRaises(ValueError):
                load_run_config(root/'experiment.yaml', root/'environment.yaml')

    def test_legacy_parser_preserves_flags_and_defaults(self):
        args = make_parser('cross_attention').parse_args(['--config', 'run.yaml', '--output_dir', 'out', '--distributed', 'false', '--local-rank', '2'])
        self.assertFalse(args.distributed)
        self.assertTrue(args.auto_resume)
        self.assertFalse(args.resume)
        self.assertEqual(args.local_rank, 2)
        self.assertEqual(args.fusion_type, 'cross_attention')


class SelectionTests(unittest.TestCase):
    def test_resume_restores_model_optimizer_scheduler_epoch_and_selection(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for name in ['original', 'lung', 'heart', 'original_m', 'lung_m', 'heart_m']:
                    setattr(self, 'visual_encoder_' + name, torch.nn.Linear(1, 1))

        with tempfile.TemporaryDirectory() as tmp:
            model = Model()
            optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            sum(p.sum() for p in model.parameters()).backward()
            optimizer.step(); scheduler.step()
            state = copy.deepcopy(model.state_dict())
            path = Path(tmp)/'last.pth'
            torch.save({'model': state, 'optimizer': optimizer.state_dict(), 'lr_scheduler': scheduler.state_dict(),
                'epoch': 4, 'best_loss': .7, 'validation_selection_state': {'best_cardiomegaly_epoch': 3}}, path)
            restored = Model()
            restored_optimizer = torch.optim.SGD(restored.parameters(), lr=.3, momentum=.9)
            restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
            epoch, loss, selection = resume_checkpoint(path, restored, restored_optimizer, restored_scheduler)
            self.assertEqual((epoch, loss, selection.best_cardiomegaly_epoch), (5, .7, 3))
            self.assertEqual(restored_optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['lr'])
            self.assertEqual(restored_scheduler.last_epoch, scheduler.last_epoch)
            self.assertTrue(restored_optimizer.state)
            for key in state:
                torch.testing.assert_allclose(restored.state_dict()[key], state[key], rtol=0, atol=0)
            torch.save({'model': state}, path)
            with self.assertRaisesRegex(ValueError, 'missing'):
                resume_checkpoint(path, restored, restored_optimizer, restored_scheduler)

    def test_checkpoint_roles_improve_independently_and_nan_is_not_selected(self):
        state = DualAUCSelectionState()
        self.assertEqual(state.consider(types.SimpleNamespace(epoch=1, cardiomegaly_auc=.8, macro_auc_stable=.7), cardiomegaly_min_delta=.01, macro_min_delta=.01), (True, True))
        self.assertEqual(state.consider(types.SimpleNamespace(epoch=2, cardiomegaly_auc=.79, macro_auc_stable=.73), cardiomegaly_min_delta=.01, macro_min_delta=.01), (False, True))
        self.assertEqual(state.best_cardiomegaly_epoch, 1)
        self.assertEqual(state.best_macro_epoch, 2)
        self.assertEqual(state.consider(types.SimpleNamespace(epoch=3, cardiomegaly_auc=float('nan'), macro_auc_stable=float('nan')), cardiomegaly_min_delta=.01, macro_min_delta=.01), (False, False))
        self.assertEqual(DualAUCSelectionState.from_dict(state.to_dict()).validations_without_improvement, 1)

    def test_last_checkpoint_cleanup_preserves_variant_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for fusion in VARIANTS:
                path = root/'checkpoint_last.pth'
                path.write_text('resume')
                cleanup_last_checkpoint(root, {'view_fusion_type': fusion})
                self.assertEqual(path.exists(), fusion in ['mean', 'gated', 'residual'])
                path.write_text('resume')
                cleanup_last_checkpoint(root, {'view_fusion_type': fusion, 'retain_last_checkpoint_after_success': True})
                self.assertTrue(path.exists())
            for role in ['cardiomegaly_auc', 'macro_auc_stable']:
                (root/f'checkpoint_best_{role}.pth').write_text('best')
            cleanup_last_checkpoint(root, {'view_fusion_type': 'mean'})
            self.assertFalse((root/'checkpoint_last.pth').exists())

    def test_architecture_metadata_distinguishes_all_six_fusions(self):
        values = [experiment_metadata({'view_fusion_type': x})['architecture'] for x in VARIANTS]
        self.assertEqual(len(set(values)), 6)
        self.assertIn('mean', experiment_metadata({'view_fusion_type': 'mean'})['architecture'].lower())
        self.assertNotIn('Transformer', experiment_metadata({'view_fusion_type': 'gated'})['architecture'])


class TokenBatch:
    def to(self, device):
        return self


class TinyTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.view_fusion = torch.nn.Linear(1, 1)
        self.temp = torch.nn.Parameter(torch.tensor(.07))

    def fusion_parameter_stats(self):
        return float(self.view_fusion.weight.detach().norm()), float(self.view_fusion.weight.detach().abs().max())

    def forward(self, original, lung, heart, text, alpha):
        x = self.view_fusion(original.flatten(1)).sum()
        return x.square(), (x-1).square() * (1+alpha), (x+1).square() * self.temp


class EpochTests(unittest.TestCase):
    def test_optimizer_metrics_and_warmup_match_each_legacy_trainer(self):
        for fusion, suffix in VARIANTS.items():
            with self.subTest(fusion=fusion):
                source = REFERENCES[f'Pretrain_A0_multiview_{suffix}_dual_auc.py']
                nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in ['train_one_epoch', 'raw_model']]
                namespace = {'torch': torch, 'utils': utils}
                exec(compile(ast.Module(body=nodes, type_ignores=[]), '<legacy epoch>', 'exec'), namespace)
                torch.manual_seed(12)
                old = TinyTrainingModel()
                new = copy.deepcopy(old)
                batch = (torch.ones(2, 1), torch.ones(2, 1), torch.ones(2, 1), ['a', 'b'])
                config = {'alpha': .4, 'view_fusion_type': fusion, 'print_freq': 100, 'debug_max_batches': 2, 'grad_clip_norm': 1.}
                results = []
                steps = []
                for model, function in [(old, namespace['train_one_epoch']), (new, train_one_epoch)]:
                    scheduler_steps = []
                    metrics = function(model, [batch, batch, batch], torch.optim.SGD(model.parameters(), lr=.01),
                        lambda *a, **k: TokenBatch(), 0, 1, torch.device('cpu'),
                        types.SimpleNamespace(step=lambda i: scheduler_steps.append(i)), config,
                        types.SimpleNamespace(distributed=False))
                    results.append(metrics); steps.append(scheduler_steps)
                self.assertEqual(steps, [[0], [0]])
                for key in results[0]:
                    self.assertAlmostEqual(results[0][key], results[1][key], places=7)
                for key in old.state_dict():
                    torch.testing.assert_allclose(old.state_dict()[key], new.state_dict()[key], rtol=0, atol=0)


class DatasetTests(unittest.TestCase):
    def test_validation_resize_supports_the_recorded_legacy_pillow_api(self):
        from torchvision import transforms
        path = ROOT/'scripts_new/vindr_multiview_classification_validation.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'build_validation_transform')
        namespace = {'transforms': transforms, 'Image': types.SimpleNamespace(BICUBIC=3),
                     'ALBEF_MEAN': (.48145466, .4578275, .40821073), 'ALBEF_STD': (.26862954, .26130258, .27577711)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<validation transform>', 'exec'), namespace)
        tensor = namespace['build_validation_transform'](8)(Image.new('RGB', (4, 4), 'white'))
        self.assertEqual(tuple(tensor.shape), (3, 8, 8))

    @classmethod
    def setUpClass(cls):
        # Load the real data module directly; unrelated legacy package bootstrap needs cv2.
        spec = importlib.util.spec_from_file_location('tested_multiview_data', ROOT/'dataset/multiview_cxr_dataset.py')
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_aligned_views_caption_and_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['lung', 'heart']:
                (root/name).mkdir()
                Image.new('L', (4, 4), 255).save(root/name/'mask.png')
            Image.new('RGB', (4, 4), 'white').save(root/'image.png')
            record = {'image': str(root/'image.png'), 'caption': 'Cardiomegaly!', 'mask_relpath': 'mask.png'}
            for name in ['lung', 'heart']:
                (root/f'{name}.json').write_text(json.dumps([record]))
            def build():
                return self.module.MultiViewCXRPretrainDataset(lung_ann_files=root/'lung.json', heart_ann_files=root/'heart.json',
                    lung_mask_root=root/'lung', heart_mask_root=root/'heart', max_words=30,
                    transform=self.module.SynchronizedCXRTransform({'image_res': 4, 'cxr_augmentation': {'enabled': False}}))
            original, lung, heart, caption = build()[0]
            self.assertEqual(caption, 'cardiomegaly')
            torch.testing.assert_allclose(original, lung); torch.testing.assert_allclose(original, heart)
            (root/'heart.json').write_text(json.dumps([{**record, 'caption': 'different'}]))
            with self.assertRaisesRegex(ValueError, 'Caption mismatch'):
                build()
            (root/'heart.json').write_text(json.dumps([record]))
            Image.new('L', (3, 3), 255).save(root/'heart/mask.png')
            with self.assertRaisesRegex(ValueError, 'size mismatch'):
                build()[0]


if __name__ == '__main__':
    unittest.main()
