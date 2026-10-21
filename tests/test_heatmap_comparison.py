import tempfile
import unittest
from pathlib import Path

import torch

from scripts_new.compare_cross_attention_heatmaps import compare_payloads


def payload():
    return {
        '__metadata__': {'image_id': 'aa', 'image_sha256': 'a', 'lung_mask_sha256': 'b', 'heart_mask_sha256': 'c',
                         'method': 'cross_attention', 'vit_layer_index': 10, 'image_res': 256,
                         'fusion_type': 'anchored', 'temperature': .07, 'uses_itm': False},
        'Cardiomegaly': {
            'positive_prompt': 'Cardiomegaly', 'negative_prompt': 'no Cardiomegaly',
            'ground_truth': 1., 'positive_similarity': .2, 'negative_similarity': .1,
            'margin': .1, 'classification_logit': 1., 'positive_probability': .73,
            **{b: {k: torch.ones(2, 2) for k in ['cam_signed_raw', 'cam_positive_raw', 'cam_vis', 'cam_vis_up']}
               for b in ['original', 'lung', 'heart', 'fused']}
        }
    }


class ComparisonTests(unittest.TestCase):
    def test_equal_outputs_pass_and_paths_may_differ(self):
        a, b = payload(), payload()
        a['__metadata__']['checkpoint'] = '/old/a'
        b['__metadata__']['checkpoint'] = '/new/a'
        report = compare_payloads(a, b)
        self.assertTrue(report['passed'])
        self.assertEqual(report['max_absolute_difference'], 0.)

    def test_raw_cam_difference_fails_even_if_visualization_is_same(self):
        a, b = payload(), payload()
        b['Cardiomegaly']['lung']['cam_signed_raw'][0, 0] = 2
        report = compare_payloads(a, b)
        self.assertFalse(report['passed'])
        self.assertTrue(any('cam_signed_raw' in x for x in report['failures']))

    def test_missing_branch_wrong_shape_and_nan_fail(self):
        for mutation in ['missing', 'shape', 'nan']:
            with self.subTest(mutation=mutation):
                a, b = payload(), payload()
                if mutation == 'missing':
                    del b['Cardiomegaly']['heart']
                elif mutation == 'shape':
                    b['Cardiomegaly']['heart']['cam_vis'] = torch.ones(1)
                else:
                    b['Cardiomegaly']['margin'] = float('nan')
                self.assertFalse(compare_payloads(a, b)['passed'])

    def test_changed_input_or_prompt_fails(self):
        for field in ['image', 'prompt']:
            a, b = payload(), payload()
            if field == 'image':
                b['__metadata__']['image_sha256'] = 'different'
            else:
                b['Cardiomegaly']['positive_prompt'] = 'different'
            self.assertFalse(compare_payloads(a, b)['passed'])

    def test_identically_incomplete_outputs_do_not_count_as_equivalent(self):
        self.assertFalse(compare_payloads({}, {})['passed'])
        a, b = payload(), payload()
        del a['Cardiomegaly']['lung']['cam_signed_raw']
        del b['Cardiomegaly']['lung']['cam_signed_raw']
        self.assertFalse(compare_payloads(a, b)['passed'])


if __name__ == '__main__':
    unittest.main()
