import ast
from pathlib import Path
import tempfile
import types
from unittest.mock import patch
import unittest
import torch

from training.anatomy_guidance import build_support_weights_from_captions, build_support_weights_a6
from training.patch_refinement import weighted_mean_a5, weighted_mean_a6, patch_identity_loss


class HelperTests(unittest.TestCase):
    def test_a4_teacher_checkpoint_interpolates_position_embeddings(self):
        # Exercise the real interpolation routine without importing optional timm.
        tree = ast.parse(Path('models/vit.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'interpolate_pos_embed')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'models/vit.py', 'exec'), namespace)
        vit = types.ModuleType('models.vit')
        vit.interpolate_pos_embed = namespace['interpolate_pos_embed']
        model = torch.nn.Module()
        model.visual_encoder = torch.nn.Module()
        model.visual_encoder.pos_embed = torch.nn.Parameter(torch.zeros(1, 5, 2))
        model.visual_encoder.patch_embed = types.SimpleNamespace(num_patches=4)
        from training.anatomy_guidance import load_state_dict_flexible
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', {'models.vit': vit}):
            checkpoint = Path(directory) / 'teacher.pth'
            torch.save({'model': {'visual_encoder.pos_embed': torch.ones(1, 2, 2)}}, checkpoint)
            load_state_dict_flexible(model, str(checkpoint))
        torch.testing.assert_allclose(model.visual_encoder.pos_embed, torch.ones(1, 5, 2))

    def test_support_modes_and_a6_default_remain_distinct(self):
        text = ['cardiomegaly', 'uncertain cardiomegaly', 'no finding']
        config = {'anatomy_target_phrase': 'cardiomegaly'}
        torch.testing.assert_allclose(build_support_weights_from_captions(text, config, 'cpu'), torch.tensor([1., 1., 0.]))
        torch.testing.assert_allclose(build_support_weights_a6(text, config, 'cpu'), torch.tensor([1., 0., 0.]))
        torch.testing.assert_allclose(build_support_weights_from_captions(text, {**config, 'support_mode': 'uncertainty_weighted'}, 'cpu'), torch.tensor([1., .5, 0.]))

    def test_fractional_weight_denominators_are_not_accidentally_unified(self):
        values = torch.tensor([2., 4.])
        weights = torch.tensor([.1, .1])
        self.assertAlmostEqual(float(weighted_mean_a5(values, weights)), .6, places=6)
        self.assertAlmostEqual(float(weighted_mean_a6(values, weights)), 3., places=6)
        self.assertEqual(float(weighted_mean_a5(values, torch.zeros(2))), 0.)
        self.assertEqual(float(weighted_mean_a6(values, torch.zeros(2))), 0.)

    def test_identity_loss_preserves_matching_maps_and_inactive_samples(self):
        target = torch.tensor([[.2, .8], [.1, .9]])
        self.assertEqual(float(patch_identity_loss(target, target, torch.ones(2))), 0.)
        prediction = target.clone(); prediction[0] = torch.tensor([.9, .1])
        self.assertEqual(float(patch_identity_loss(prediction, target, torch.tensor([0., 1.]))), 0.)


if __name__ == '__main__':
    unittest.main()
