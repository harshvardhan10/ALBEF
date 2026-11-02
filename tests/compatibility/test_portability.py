"""Opt-in asset IO and local-config safeguards; no private assets or downloads."""
import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
HELPERS = ('scripts.src', 'scripts_new.src')


class PortabilityTests(unittest.TestCase):
    def helper(self, name):
        module = importlib.import_module(name)
        self.assertTrue(hasattr(module, '_resolve_pretrained_sources'),
                        'Explicit asset resolver is missing')
        return module

    def historical_tokenizer(self):
        source = ROOT / 'archive/source_snapshots/evaluation/legacy_src.py'
        self.assertTrue(source.is_file(), 'Historical IO snapshot is missing')
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'build_model_and_tokenizer')
        call = next(n for n in ast.walk(function) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == 'BertTokenizer')
        return ast.literal_eval(call.args[0])

    def test_asset_defaults_and_explicit_environment_precedence(self):
        for name in HELPERS:
            with self.subTest(helper=name), patch.dict(os.environ):
                os.environ.pop('ALBEF_TOKENIZER_PATH', None)
                os.environ.pop('ALBEF_TEXT_ENCODER', None)
                module = self.helper(name)
                self.assertEqual(module._resolve_pretrained_sources(),
                                 (self.historical_tokenizer(), 'bert-base-uncased'))
                os.environ['ALBEF_TOKENIZER_PATH'] = '/local/tokenizer'
                os.environ['ALBEF_TEXT_ENCODER'] = '/local/encoder'
                self.assertEqual(module._resolve_pretrained_sources(),
                                 ('/local/tokenizer', '/local/encoder'))
                self.assertEqual(module._resolve_pretrained_sources(
                    tokenizer_path=Path('/explicit/tokenizer'),
                    text_encoder='/explicit/encoder'),
                    ('/explicit/tokenizer', '/explicit/encoder'))
                with self.assertRaises(ValueError):
                    module._resolve_pretrained_sources(tokenizer_path='')
                os.environ['ALBEF_TEXT_ENCODER'] = ''
                with self.assertRaises(ValueError):
                    module._resolve_pretrained_sources()

    def test_real_local_tokenizer_and_checkpoint_io(self):
        for name in HELPERS:
            with self.subTest(helper=name), tempfile.TemporaryDirectory() as temporary:
                module = self.helper(name)
                root = Path(temporary)
                tokenizer = root / 'tokenizer'
                tokenizer.mkdir()
                (tokenizer / 'vocab.txt').write_text(
                    '[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nchest\n')
                config = root / 'config.yaml'
                config.write_text('image_res: 32\n')
                checkpoint = root / 'checkpoint.pth'
                torch.save({'model': {'weight': torch.tensor([[0.75]]),
                                      'bias': torch.tensor([0.25])}}, checkpoint)
                def construct(**kwargs):
                    self.assertEqual(kwargs['text_encoder'], '/explicit/encoder')
                    self.assertEqual(kwargs['tokenizer'].encode(
                        'A chest', add_special_tokens=False), [5, 6])
                    self.assertFalse(kwargs['init_deit'])
                    return torch.nn.Linear(1, 1)
                # The unavailable pretrained model constructor alone is isolated;
                # tokenizer file loading, tokenization, torch.load/state loading,
                # device handling and eval mode execute normally.
                with patch.object(module, 'ALBEF', side_effect=construct):
                    model, loaded, cfg, device = module.build_model_and_tokenizer(
                        config, checkpoint, 'cpu', tokenizer_path=tokenizer,
                        text_encoder='/explicit/encoder')
                self.assertEqual(model.weight.item(), 0.75)
                self.assertEqual(model.bias.item(), 0.25)
                self.assertFalse(model.training)
                self.assertEqual(str(device), 'cpu')
                self.assertEqual(cfg, {'image_res': 32})
                self.assertEqual(loaded.encode('a chest', add_special_tokens=False), [5, 6])

    def test_original_functions_and_loader_body_remain_equivalent(self):
        for family in ('legacy', 'modern'):
            with self.subTest(family=family):
                old_path = ROOT / ('archive/source_snapshots/evaluation/' + family + '_src.py')
                self.assertTrue(old_path.is_file(), 'Original helper snapshot is missing')
                old = ast.parse(old_path.read_text())
                current = ast.parse((ROOT / ('src/cxr_vlm/evaluation/' + family + '/src.py')).read_text())
                old_defs = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
                new_defs = {n.name: n for n in current.body if isinstance(n, ast.FunctionDef)}
                self.assertEqual(set(new_defs) - set(old_defs), {'_resolve_pretrained_sources'})
                for function, expected in old_defs.items():
                    actual = copy.deepcopy(new_defs[function])
                    if function == 'build_model_and_tokenizer':
                        self.assertEqual([a.arg for a in actual.args.kwonlyargs],
                                         ['tokenizer_path', 'text_encoder'])
                        self.assertTrue(all(isinstance(d, ast.Constant) and d.value is None
                                            for d in actual.args.kw_defaults))
                        actual.args.kwonlyargs = []
                        actual.args.kw_defaults = []
                        actual.body = [n for n in actual.body if not (
                            isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                            and isinstance(n.value.func, ast.Name)
                            and n.value.func.id == '_resolve_pretrained_sources')]
                        tokenizer_default = self.historical_tokenizer()
                        class Restore(ast.NodeTransformer):
                            def visit_Name(self, node):
                                if node.id == 'tokenizer_source':
                                    return ast.Constant(value=tokenizer_default)
                                if node.id == 'text_encoder_source':
                                    return ast.Constant(value='bert-base-uncased')
                                return node
                        actual = Restore().visit(actual)
                    self.assertEqual(ast.dump(expected), ast.dump(actual), function)

    def config_tool(self):
        self.assertTrue((ROOT / 'tools/prepare_config.py').is_file(),
                        'Local config generator is missing')
        return importlib.import_module('tools.prepare_config')

    def config_fixture(self, root):
        data = {'train_file': ['/old/train.json'], 'bert_config': '/old/bert.json',
                'image_res': 256, 'lambda_support': 0.01, 'mask_root': None,
                'optimizer': {'lr': 0.0001},
                'vindr_validation': {'labels_csv': '/old/labels.csv', 'enabled': True}}
        source = root / 'historical.yaml'
        source.write_text(yaml.safe_dump(data))
        return source, data

    def test_local_config_rebinds_paths_only_and_preserves_source(self):
        tool = self.config_tool()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, original = self.config_fixture(root)
            before = source.read_bytes()
            overrides = {'train_file': [str(root / 'train.json')],
                         'bert_config': str(root / 'bert.json'),
                         'vindr_validation.labels_csv': str(root / 'labels.csv')}
            output, provenance = tool.prepare_config(source, root / 'new-run', overrides)
            actual = yaml.safe_load(output.read_text())
            expected = copy.deepcopy(original)
            expected['train_file'] = overrides['train_file']
            expected['bert_config'] = overrides['bert_config']
            expected['vindr_validation']['labels_csv'] = overrides['vindr_validation.labels_csv']
            self.assertEqual(actual, expected)
            self.assertEqual(source.read_bytes(), before)
            record = json.loads(provenance.read_text())
            self.assertEqual(record['source_sha256'], hashlib.sha256(before).hexdigest())
            self.assertEqual(record['config_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertFalse(record['assets_verified'])
            with self.assertRaises((ValueError, FileExistsError)):
                tool.prepare_config(source, root / 'new-run', overrides)
            self.assertEqual(yaml.safe_load(output.read_text()), expected)

    def test_config_rejects_semantic_changes_inactive_fields_and_list_shape(self):
        tool = self.config_tool()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, original = self.config_fixture(root)
            invalid = [{'image_res': '/fake'}, {'optimizer.lr': '/fake'},
                       {'lambda_support': '/fake'}, {'mask_root': '/fake'},
                       {'train_file': '/fake'}, {'train_file': ['/a', '/b']},
                       {'train_file': []}, {'bert_config': 'relative.json'},
                       {'missing.labels_csv': '/fake'}]
            for i, overrides in enumerate(invalid):
                output = root / ('invalid-' + str(i))
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    tool.prepare_config(source, output, overrides)
                self.assertFalse(output.exists())
            self.assertEqual(yaml.safe_load(source.read_text()), original)
            with self.assertRaises(ValueError):
                tool.prepare_config(source, ROOT / 'unwanted-generated-config',
                                    {'bert_config': str(root / 'bert.json')})

    def test_config_refuses_existing_dangling_output_symlink(self):
        tool = self.config_tool()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.config_fixture(root)
            target = root / 'not-created'
            alias = root / 'existing-alias'
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                tool.prepare_config(source, alias, {'bert_config': str(root / 'bert.json')})
            self.assertTrue(alias.is_symlink())
            self.assertFalse(target.exists())

    def test_config_path_binding_does_not_mutate_another_yaml_alias(self):
        tool = self.config_tool()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'aliases.yaml'
            source.write_text('first: &shared\n  labels_csv: /old/labels.csv\nsecond: *shared\n')
            output, _ = tool.prepare_config(source, root / 'new',
                                             {'first.labels_csv': str(root / 'new.csv')})
            actual = yaml.safe_load(output.read_text())
            self.assertEqual(actual['second']['labels_csv'], '/old/labels.csv')
            self.assertEqual(actual['first']['labels_csv'], str(root / 'new.csv'))

    def test_config_cli_rejects_duplicate_path_assignments(self):
        tool = self.config_tool()
        with self.assertRaises(ValueError):
            tool.load_overrides(None, ['bert_config=/a', 'bert_config=/b'])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'duplicate.json'
            path.write_text('{"bert_config":"/a","bert_config":"/b"}')
            with self.assertRaises(ValueError):
                tool.load_overrides(path, [])


if __name__ == '__main__':
    unittest.main()
