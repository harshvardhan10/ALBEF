import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from scripts_new.render_cross_attention_heatmaps import render_case


class OverlayTests(unittest.TestCase):
    def test_render_original_and_two_labels_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (4, 4), (100, 100, 100)).save(root/'aa.png')
            payload = {'__metadata__': {'image_path': str(root/'aa.png'), 'image_id': 'aa'}}
            for label in ['Cardiomegaly', 'Pleural effusion']:
                payload[label] = {'fused': {'cam_vis_up': torch.ones(4, 4)}}
            torch.save(payload, root/'aa.pt')
            render_case(root/'aa.pt', root/'overlay.png')
            with Image.open(root/'overlay.png') as overlay:
                self.assertEqual(overlay.size, (768, 280))
                self.assertEqual(overlay.mode, 'RGB')
            with self.assertRaises(FileExistsError):
                render_case(root/'aa.pt', root/'overlay.png')


if __name__ == '__main__':
    unittest.main()
