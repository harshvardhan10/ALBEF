"""CPU behavior checks; no checkpoint download or research-stack import."""
import tempfile
import unittest
import ast
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from scripts_new import extract_cross_attention_fusion_itc_margin_gradcam_heatmaps as heatmaps
from scripts_new.compare_cross_attention_heatmaps import compare_directories


class HeatmapTests(unittest.TestCase):
    def test_extraction_writes_complete_comparable_artifacts_through_real_fusion(self):
        # Only replace the unavailable heavyweight model/tokenizer assembly.
        # Execute the actual anchored-fusion classes from the untouched source.
        tree = ast.parse((heatmaps.REPOSITORY_ROOT/'models/model_pretrain_multiview_cross_attention_fusion.py').read_text())
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in {'AnchoredCrossAttentionLayer', 'ViewCrossAttentionFusion'}]
        namespace = {'torch': torch, 'nn': torch.nn}
        exec(compile(ast.Module(body=classes, type_ignores=[]), 'legacy_fusion', 'exec'), namespace)

        class TinyVision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(3, 8)
                self.blocks = torch.nn.ModuleList([torch.nn.Module()])
                self.blocks[0].attn = torch.nn.Module()
                self.blocks[0].attn.attn_drop = torch.nn.Dropout(0.)

            def forward(self, image):
                tokens = image.flatten(2).transpose(1, 2)
                tokens = torch.cat([tokens.mean(1, keepdim=True), tokens], 1)
                tokens = self.projection(tokens)
                attn = (tokens @ tokens.transpose(-1, -2)).unsqueeze(1).softmax(-1)
                attn = self.blocks[0].attn.attn_drop(attn)
                return attn[:, 0] @ tokens

        class TinyBert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 8)

            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for view in ['original', 'lung', 'heart']:
                    setattr(self, 'visual_encoder_' + view, TinyVision())
                self.view_fusion = namespace['ViewCrossAttentionFusion'](hidden_dim=8, num_heads=2, mlp_ratio=2.)
                self.text_encoder = torch.nn.Module()
                self.text_encoder.bert = TinyBert()
                self.text_proj = torch.nn.Linear(8, 4)
                self.vision_proj = torch.nn.Linear(8, 4)
                self.temp = torch.nn.Parameter(torch.tensor(.07))

            def _fuse_tokens(self, *views, momentum=False):
                return self.view_fusion(*views)

        def tokenizer(prompts, **kwargs):
            ids = torch.arange(len(prompts)).reshape(-1, 1)
            return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for d in ['images', 'lung/aa', 'heart/aa', 'tokenizer']:
                (root/d).mkdir(parents=True)
            Image.fromarray(np.arange(12, dtype=np.uint8).reshape(2, 2, 3) * 20).save(root/'images/aa.png')
            Image.new('L', (2, 2), 255).save(root/'lung/aa/aa.png')
            Image.new('L', (2, 2), 255).save(root/'heart/aa/aa.png')
            (root/'cases.csv').write_text('image_id,Cardiomegaly,Pleural effusion\naa,1,0\n')
            (root/'config.yaml').write_text('image_res: 2\n')
            (root/'tokenizer/vocab.txt').write_text('[PAD]\n')
            model = TinyModel().eval()
            torch.save({'model': model.state_dict()}, root/'weights.pth')
            args = argparse.Namespace(config=str(root/'config.yaml'), checkpoint=str(root/'weights.pth'),
                labels_csv=str(root/'cases.csv'), images_root=str(root/'images'), lung_mask_root=str(root/'lung'),
                heart_mask_root=str(root/'heart'), output_dir=str(root/'baseline'),
                target_labels=['Cardiomegaly', 'Pleural effusion'], max_images=None, positive_only_label=None,
                max_text_length=32, device='cpu', text_encoder='local', tokenizer_name_or_path=str(root/'tokenizer'),
                temperature=None, vit_layer=-1, checkpoint_role='best_cardiomegaly_auc', training_commit='unconfirmed')
            with patch.object(heatmaps, 'build_model_and_tokenizer', return_value=(model, tokenizer, {'image_res': 2}, torch.device('cpu'))):
                heatmaps.extract(args)
                args.output_dir = str(root/'repeat')
                heatmaps.extract(args)
            output = torch.load(root/'baseline/aa.pt')
            self.assertEqual(set(output), {'__metadata__', 'Cardiomegaly', 'Pleural effusion'})
            self.assertEqual(output['__metadata__']['vit_layer_index'], 0)
            for label in args.target_labels:
                for view in ['original', 'lung', 'heart', 'fused']:
                    self.assertEqual(tuple(output[label][view]['cam_vis'].shape), (2, 2))
                    self.assertTrue(torch.isfinite(output[label][view]['cam_signed_raw']).all())
            manifest = json.loads((root/'baseline/cross_attention_fusion_itc_margin_gradcam_manifest.json').read_text())
            self.assertEqual(manifest['case_ids'], ['aa'])
            self.assertEqual(manifest['anchor_view'], 'original')
            report = compare_directories(root/'baseline', root/'repeat')
            self.assertTrue(report['passed'], report['failures'])
            for view in ['original', 'lung', 'heart']:
                self.assertFalse(getattr(model, 'visual_encoder_' + view).blocks[0].attn.attn_drop._forward_hooks)

    def test_cam_uses_per_head_relu_before_averaging(self):
        attention = torch.ones(1, 2, 5, 5, requires_grad=True)
        weights = torch.zeros_like(attention)
        weights[0, 0, 0, 1:] = torch.tensor([2., -2., 0., 4.])
        weights[0, 1, 0, 1:] = torch.tensor([-4., 4., 0., 2.])
        (attention * weights).sum().backward()
        cam = heatmaps.compute_gradcam(SimpleNamespace(attention_map=attention))
        np.testing.assert_allclose(cam['cam_signed_raw'], [[-1., 1.], [0., 3.]])
        np.testing.assert_allclose(cam['cam_positive_raw'], [[1., 2.], [0., 3.]])
        np.testing.assert_allclose(cam['cam_vis'], [[1/3, 2/3], [0., 1.]])

    def test_zero_cam_is_finite(self):
        a = torch.ones(1, 1, 5, 5, requires_grad=True)
        (a * 0).sum().backward()
        cam = heatmaps.compute_gradcam(SimpleNamespace(attention_map=a))
        self.assertEqual(float(cam['cam_vis'].sum()), 0.)
        self.assertTrue(torch.isfinite(cam['cam_vis']).all())

    def test_nonfinite_attention_fails(self):
        a = torch.full((1, 1, 5, 5), float('nan'), requires_grad=True)
        a.sum().backward()
        with self.assertRaisesRegex(ValueError, 'finite'):
            heatmaps.compute_gradcam(SimpleNamespace(attention_map=a))

    def test_missing_gradients_fail(self):
        with self.assertRaisesRegex(RuntimeError, 'gradient'):
            heatmaps.compute_gradcam(SimpleNamespace(attention_map=torch.ones(1, 1, 5, 5)))

    def test_hook_cleanup_after_exception(self):
        module = torch.nn.Module()
        module.attn_drop = torch.nn.Dropout(0.)
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with heatmaps.VisualAttentionCapture(module):
                raise RuntimeError('sentinel')
        self.assertFalse(module.attn_drop._forward_hooks)

    def test_checkpoint_wrong_keys_fail_strictly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.pth'
            torch.save({'model': {'wrong': torch.ones(2)}}, path)
            with self.assertRaises(RuntimeError):
                heatmaps.load_fused_checkpoint(torch.nn.Linear(2, 2), path)

    def test_ddp_prefix_is_removed_and_checkpoint_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 2)
            path = Path(tmp)/'good.pth'
            torch.save({'model': {'module.weight': torch.ones(2, 2), 'module.bias': torch.zeros(2)}}, path)
            heatmaps.load_fused_checkpoint(model, path)
            np.testing.assert_allclose(model(torch.ones(1, 2)).detach(), [[2., 2.]])

    def test_selection_preserves_order_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp)/'cases.csv'
            csv.write_text('image_id,Cardiomegaly\nbb,1\naa,0\n')
            data, column = heatmaps.load_selection(csv, ['Cardiomegaly'], 2, None)
            self.assertEqual(data[column].tolist(), ['bb', 'aa'])
            csv.write_text('image_id,Cardiomegaly\naa,1\naa,0\n')
            with self.assertRaisesRegex(ValueError, 'unique'):
                heatmaps.load_selection(csv, ['Cardiomegaly'], None, None)

    def test_selection_rejects_path_traversal_and_missing_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp)/'cases.csv'
            csv.write_text('image_id,Cardiomegaly\n../bad,1\n')
            with self.assertRaises(ValueError):
                heatmaps.load_selection(csv, ['Cardiomegaly'], None, None)
            with self.assertRaises(ValueError):
                heatmaps.load_selection(csv, ['Pleural effusion'], None, None)

    def test_compositing_and_mask_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for d in ['images', 'lung/aa', 'heart/aa']:
                (root/d).mkdir(parents=True)
            Image.new('RGB', (2, 2), (100, 100, 100)).save(root/'images/aa.png')
            Image.new('L', (2, 2), 0).save(root/'lung/aa/aa.png')
            Image.new('L', (2, 2), 255).save(root/'heart/aa/aa.png')
            original, lung, heart, *_ = heatmaps.load_three_view_images(root/'images', 'aa', root/'lung', root/'heart')
            self.assertEqual(np.asarray(lung).sum(), 0)
            np.testing.assert_array_equal(heart, original)
            Image.new('L', (3, 3), 255).save(root/'heart/aa/aa.png')
            with self.assertRaisesRegex(ValueError, 'Size mismatch'):
                heatmaps.load_three_view_images(root/'images', 'aa', root/'lung', root/'heart')
            (root/'heart/aa/aa.png').unlink()
            with self.assertRaises(FileNotFoundError):
                heatmaps.load_three_view_images(root/'images', 'aa', root/'lung', root/'heart')

    def test_output_directory_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'results'
            heatmaps.prepare_output_directory(path)
            with self.assertRaises(FileExistsError):
                heatmaps.prepare_output_directory(path)

    def test_layer_and_prompt_contract(self):
        self.assertEqual(heatmaps.resolve_layer_index(12, -2), 10)
        self.assertEqual(heatmaps.build_bare_prompt_pair('Pleural effusion'), ('Pleural effusion', 'no Pleural effusion'))
        with self.assertRaises(ValueError):
            heatmaps.resolve_layer_index(12, -13)


if __name__ == '__main__':
    unittest.main()
