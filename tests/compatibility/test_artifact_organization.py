"""Preservation and execution-context checks; never execute research notebooks."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ArtifactOrganizationTests(unittest.TestCase):
    def catalog(self, name):
        path = ROOT / 'docs/audit' / name
        self.assertTrue(path.is_file(), 'Missing artifact catalog: ' + name)
        with path.open() as handle:
            return list(csv.DictReader(handle))

    def verify_rows(self, rows):
        self.assertEqual(len(rows), len({r['legacy_path'] for r in rows}))
        self.assertEqual(len(rows), len({r['canonical_path'] for r in rows}))
        for row in rows:
            with self.subTest(path=row['legacy_path']):
                old = ROOT / row['legacy_path']
                canonical = ROOT / row['canonical_path']
                self.assertEqual(hashlib.sha256(canonical.read_bytes()).hexdigest(),
                                 row['original_sha256'])
                if row['status'] == 'moved':
                    self.assertTrue(old.is_symlink())
                    self.assertEqual(old.resolve(), canonical.resolve())
                else:
                    self.assertEqual(old, canonical)
                    self.assertFalse(old.is_symlink())

    def test_notebooks_preserve_original_bytes_and_safe_move_scope(self):
        rows = self.catalog('notebook_catalog.csv')
        self.assertEqual(len(rows), 28)
        self.assertEqual(sum(r['status'] == 'moved' for r in rows), 7)
        self.verify_rows(rows)
        for row in rows:
            with self.subTest(notebook=row['legacy_path']):
                self.assertEqual(row['legacy_working_directory'],
                                 str(Path(row['legacy_path']).parent))
                if row['status'] == 'moved':
                    notebook = json.loads((ROOT / row['canonical_path']).read_text())
                    for cell in notebook['cells']:
                        self.assertFalse(cell.get('outputs', []))
                        self.assertFalse(cell.get('attachments', {}))
                        self.assertNotIn('data:image/', ''.join(cell.get('source', [])))

    def test_results_preserve_aggregate_and_held_prediction_files(self):
        rows = self.catalog('result_catalog.csv')
        self.assertEqual(len(rows), 36)
        self.assertEqual(sum(r['status'] == 'moved' for r in rows), 34)
        self.assertEqual(sum(r['status'] == 'held_predictions' for r in rows), 2)
        self.verify_rows(rows)
        for row in rows:
            if row['status'] == 'moved' and row['canonical_path'].endswith('.csv'):
                with (ROOT / row['canonical_path']).open() as handle:
                    columns = next(csv.reader(handle))
                self.assertNotIn('image_id', columns)
                self.assertNotIn('patient_id', columns)

    def inspect_context(self, name):
        tool = ROOT / 'tools/notebook_context.py'
        self.assertTrue(tool.is_file(), 'Missing notebook context inspector')
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        return subprocess.run([sys.executable, '-B', str(tool), name],
                              cwd=ROOT, env=env, capture_output=True, text=True,
                              timeout=30)

    def test_context_preserves_historical_working_directory(self):
        legacy = 'notebooks/localization_pointing_game.ipynb'
        canonical = 'notebooks/evaluation/localization_pointing_game.ipynb'
        a = self.inspect_context(legacy)
        b = self.inspect_context(canonical)
        self.assertEqual(a.returncode, 0, a.stderr)
        self.assertEqual(b.returncode, 0, b.stderr)
        self.assertEqual(a.stdout, b.stdout)
        context = json.loads(a.stdout)
        self.assertEqual(context['kernel_working_directory'], str(ROOT / 'notebooks'))
        self.assertEqual(context['open_legacy_path'], str(ROOT / legacy))
        self.assertEqual(context['canonical_path'], str(ROOT / canonical))
        self.assertFalse(context['executed'])
        # The upstream root notebook must retain a root cwd, not notebooks/.
        root = self.inspect_context('visualization.ipynb')
        self.assertEqual(root.returncode, 0, root.stderr)
        self.assertEqual(json.loads(root.stdout)['kernel_working_directory'], str(ROOT))

    def test_context_rejects_unknown_notebook(self):
        result = self.inspect_context('../unknown.ipynb')
        self.assertEqual(result.returncode, 2)
        self.assertIn('registered', result.stderr)


if __name__ == '__main__':
    unittest.main()
