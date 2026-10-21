import gc
import importlib
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.fusion.factory import build_fusion_module
from tests.multiview_support import (
    REFERENCES, ROOT, TOKENIZER, VARIANTS, TinyBertConfig, TinyMaskedLM, TinyVision,
    config, legacy_model,
)


class FusionTests(unittest.TestCase):
    def test_extracted_fusions_match_frozen_outputs_and_input_gradients(self):
        classes = {'mean': 'SimpleMeanFusion', 'gated': 'ImageLevelGatedFusion',
                   'residual': 'OriginalAnchoredResidualFusion', 'transformer': 'ViewTransformerFusion',
                   'cross_attention': 'ViewCrossAttentionFusion'}
        for variant, name in classes.items():
            with self.subTest(variant=variant):
                kwargs = {} if variant == 'mean' else {'hidden_dim': 8}
                if variant in ['transformer', 'cross_attention']:
                    kwargs.update(num_heads=2, num_layers=2, mlp_ratio=2., dropout=0.)
                if variant in ['gated', 'residual']:
                    kwargs['gate_hidden_dim'] = 8
                torch.manual_seed(7)
                old = legacy_model(variant)[name](**kwargs).eval()
                new = build_fusion_module(variant, 8, {
                    'view_fusion_gate': {'hidden_dim': 8},
                    'view_fusion_residual': {'hidden_dim': 8},
                    'view_fusion_transformer': {'num_heads': 2, 'mlp_ratio': 2.},
                    'view_fusion_cross_attention': {'num_heads': 2, 'mlp_ratio': 2.},
                }).eval()
                new.load_state_dict(old.state_dict(), strict=True)
                views = tuple(torch.randn(2, 5, 8, requires_grad=True) for _ in range(3))
                other = tuple(v.detach().clone().requires_grad_() for v in views)
                expected, actual = old(*views), new(*other)
                torch.testing.assert_allclose(actual, expected)
                expected.square().sum().backward()
                actual.square().sum().backward()
                for a, b in zip(views, other):
                    torch.testing.assert_allclose(a.grad, b.grad)

    def test_factory_rejects_unknown_and_cross_attention_view_reordering(self):
        with self.assertRaises(ValueError):
            build_fusion_module('unknown', 8, {})
        with self.assertRaises(ValueError):
            build_fusion_module('cross_attention', 8, {'view_fusion_cross_attention': {'memory_views': ['heart', 'lung']}})


class SharedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only replace encoder assembly; all fusion/objective/state code is production code.
        vit = types.ModuleType('models.vit')
        vit.VisionTransformer = TinyVision
        vit.interpolate_pos_embed = lambda state, encoder: state
        bert = types.ModuleType('models.xbert')
        bert.BertConfig = TinyBertConfig
        bert.BertForMaskedLM = TinyMaskedLM
        cls.dependencies = patch.dict(sys.modules, {'models.vit': vit, 'models.xbert': bert})
        cls.dependencies.start()
        cls.shared = importlib.import_module('models.model_pretrain_multiview')
        cls.gather = patch.object(cls.shared, 'concat_all_gather', lambda value: value)
        cls.gather.start()

    @classmethod
    def tearDownClass(cls):
        cls.gather.stop()
        cls.dependencies.stop()
        # Do not leave synthetic encoder modules bound for other tests.
        for name in list(sys.modules):
            if name == 'models.model_pretrain_multiview' or name.startswith('models.model_pretrain_multiview_'):
                sys.modules.pop(name, None)

    def test_all_six_wrappers_preserve_state_initialization_features_losses_and_gradients(self):
        for variant, suffix in VARIANTS.items():
            with self.subTest(variant=variant):
                torch.manual_seed(41)
                old = legacy_model(variant)['ALBEF'](config=config(variant), tokenizer=TOKENIZER, init_deit=False)
                torch.manual_seed(41)
                module = importlib.import_module('models.model_pretrain_multiview_' + suffix)
                new = module.ALBEF(config=config(variant), tokenizer=TOKENIZER, init_deit=False)
                old_state, new_state = old.state_dict(), new.state_dict()
                self.assertEqual(set(old_state), set(new_state))
                for key in old_state:
                    torch.testing.assert_allclose(new_state[key], old_state[key], rtol=0., atol=0.)
                new.load_state_dict(old_state, strict=True)
                torch.manual_seed(55)
                views = tuple(torch.randn(2, 3, 2, 2) for _ in range(3))
                old.eval(); new.eval()
                a = old.encode_image_views(*views, momentum=False)
                b = new.encode_image_views(*views, momentum=False)
                for expected, actual in zip(a, b):
                    torch.testing.assert_allclose(actual, expected)
                text = types.SimpleNamespace(input_ids=torch.tensor([[1, 3, 4], [1, 5, 6]]), attention_mask=torch.ones(2, 3, dtype=torch.long))
                old.train(); new.train()
                torch.manual_seed(71)
                old_losses = old(*views, text, alpha=.4)
                torch.manual_seed(71)
                new_losses = new(*views, text, alpha=.4)
                for expected, actual in zip(old_losses, new_losses):
                    torch.testing.assert_allclose(actual, expected)
                sum(old_losses).backward(); sum(new_losses).backward()
                for (key, expected), (new_key, actual) in zip(old.named_parameters(), new.named_parameters()):
                    self.assertEqual(key, new_key)
                    self.assertEqual(expected.grad is None, actual.grad is None, key)
                    if expected.grad is not None:
                        torch.testing.assert_allclose(actual.grad, expected.grad)
                for key in ['image_queue', 'text_queue', 'queue_ptr']:
                    torch.testing.assert_allclose(new.state_dict()[key], old.state_dict()[key])
                for online, momentum in new.model_pairs:
                    self.assertIsNot(online, momentum)
                    for value in momentum.parameters():
                        self.assertFalse(value.requires_grad)
                        self.assertIsNone(value.grad)
                del old, new, old_state, new_state, a, b, old_losses, new_losses
                gc.collect()

    def test_fc_keeps_separate_normalization_keys_and_always_checks_nonfinite_values(self):
        module = importlib.import_module('models.model_pretrain_multiview_fusion')
        model = module.ALBEF(config=config('fc'), tokenizer=TOKENIZER, init_deit=False)
        self.assertIn('view_fusion.weight', model.state_dict())
        self.assertIn('view_fusion_norm.weight', model.state_dict())
        with self.assertRaises(FloatingPointError):
            model._check_finite('test', torch.tensor([float('nan')]))

    def test_shared_model_requires_an_explicit_or_wrapper_fusion(self):
        with self.assertRaises(ValueError):
            self.shared.ALBEF(config=config('mean'), tokenizer=TOKENIZER, init_deit=False)


if __name__ == '__main__':
    unittest.main()
