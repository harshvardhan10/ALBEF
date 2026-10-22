import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluation.prompts import build_prompts_for_label
from evaluation.experiment_metadata import write_score_metadata
from evaluation.artifacts import atomic_save_scores_npz


class MetadataTests(unittest.TestCase):
    def test_atomic_score_writer_preserves_arrays_and_writes_selected_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'selected.npz'
            outputs = {'image_ids': ['aa'], 'label_names': ['Cardiomegaly', 'Pleural effusion'], 'y_true': [[1, 0]], 'scores': [[.8, .2]]}
            metrics = SimpleNamespace(epoch=2, view_type='original', cardiomegaly_auc=.9,
                macro_auc_all_evaluable=.8, macro_auc_stable=.8, macro_auc=.8, cardiomegaly_best_threshold=.5)
            atomic_save_scores_npz(path, outputs=outputs, metrics=metrics, metadata={'checkpoint_role': 'best_macro_auc_stable'})
            with np.load(path, allow_pickle=True) as data:
                np.testing.assert_allclose(data['scores'], [[.8, .2]])
                self.assertEqual(data['label_names'].tolist(), ['Cardiomegaly', 'Pleural effusion'])
                self.assertEqual(data['y_true'].dtype, np.int8)
            metadata = json.loads(Path(str(path)+'.metadata.json').read_text())
            self.assertEqual(metadata['checkpoint_role'], 'best_macro_auc_stable')
            self.assertFalse(Path(str(path)+'.tmp').exists())

    def test_prompt_template_expansion_and_no_finding_special_case(self):
        self.assertEqual(build_prompts_for_label('No_finding'), ['No finding'])
        self.assertEqual(build_prompts_for_label('Pleural_effusion'), [
            'Pleural effusion', 'A chest X-ray showing Pleural effusion.',
            'Chest radiograph demonstrating Pleural effusion.', 'There is evidence of Pleural effusion.',
            'Findings are consistent with Pleural effusion.',
        ])

    def test_sidecar_records_actual_column_order_prompts_and_unknown_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'scores.npz'
            path.write_bytes(b'test artifact')
            outputs = {'label_names': ['Pleural effusion', 'Cardiomegaly'], 'image_ids': ['aa', 'bb']}
            write_score_metadata(path, outputs, SimpleNamespace(epoch=3, view_type='multiview_fusion'))
            data = json.loads(Path(str(path)+'.metadata.json').read_text())
            self.assertEqual(data['label_order'], ['Pleural effusion', 'Cardiomegaly'])
            self.assertEqual(data['num_images'], 2)
            self.assertEqual(data['prompts']['Cardiomegaly'][0], 'Cardiomegaly')
            self.assertIsNone(data['checkpoint_role'])
            self.assertEqual(data['threshold_source'], 'current validation split; not a held-out test threshold')
            self.assertEqual(len(data['artifact_sha256']), 64)
            write_score_metadata(path, outputs, SimpleNamespace(epoch=3, view_type='multiview_fusion'), {'checkpoint_role': 'best_cardiomegaly_auc'})
            selected = json.loads(Path(str(path)+'.metadata.json').read_text())
            self.assertEqual(selected['checkpoint_role'], 'best_cardiomegaly_auc')


if __name__ == '__main__':
    unittest.main()
