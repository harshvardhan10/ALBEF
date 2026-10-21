"""Small real tensor dependencies for comparing original/refactored models.

These are not substitutes for server ViT/BERT/checkpoint acceptance tests.
"""
import ast
import gzip
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
with gzip.open(ROOT/'tests/references/multiview_models_6320a2e.json.gz', 'rt') as handle:
    REFERENCES = json.load(handle)

VARIANTS = {
    'fc': 'fusion', 'mean': 'mean_fusion', 'gated': 'gated_fusion',
    'residual': 'residual_fusion', 'transformer': 'transformer_fusion',
    'cross_attention': 'cross_attention_fusion',
}


class TinyVision(nn.Module):
    def __init__(self, embed_dim=768, **kwargs):
        super().__init__()
        self.projection = nn.Linear(3, embed_dim)

    def forward(self, image):
        tokens = image.flatten(2).transpose(1, 2)
        tokens = torch.cat([tokens.mean(1, keepdim=True), tokens], dim=1)
        return self.projection(tokens)


class TinyBertConfig:
    @classmethod
    def from_json_file(cls, path):
        return SimpleNamespace(hidden_size=8, vocab_size=16)


class TinyBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.fusion = nn.Linear(768, 8)

    def forward(self, input_ids=None, encoder_embeds=None, encoder_hidden_states=None, **kwargs):
        value = self.embedding(input_ids) if encoder_embeds is None else encoder_embeds
        if encoder_hidden_states is not None:
            value = value + self.fusion(encoder_hidden_states.mean(1))[:, None, :]
        return SimpleNamespace(last_hidden_state=value)


class TinyMaskedLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TinyBertConfig.from_json_file(None)
        self.bert = TinyBert()
        self.head = nn.Linear(8, 16)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def forward(self, input_ids, labels=None, soft_labels=None, alpha=0., return_logits=False, **kwargs):
        logits = self.head(self.bert(input_ids=input_ids, **kwargs).last_hidden_state)
        if return_logits:
            return logits
        hard = F.cross_entropy(logits.reshape(-1, 16), labels.reshape(-1), ignore_index=-100)
        active = labels != -100
        soft = -(F.log_softmax(logits, -1) * soft_labels).sum(-1)[active].mean()
        return SimpleNamespace(loss=(1-alpha)*hard + alpha*soft)


def legacy_model(variant):
    source = REFERENCES['models/model_pretrain_multiview_' + VARIANTS[variant] + '.py']
    tree = ast.parse(source)
    # Model bodies/functions are unchanged; bypass only heavyweight dependency imports.
    nodes = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    namespace = dict(torch=torch, nn=nn, F=F, partial=partial, Tuple=Tuple,
                     VisionTransformer=TinyVision, BertConfig=TinyBertConfig, BertForMaskedLM=TinyMaskedLM,
                     interpolate_pos_embed=lambda state, encoder: state)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen legacy ' + variant + '>', 'exec'), namespace)
    # Single-process CPU tests cannot validate distributed gather.
    namespace['concat_all_gather'] = lambda value: value
    return namespace


def config(variant):
    return {
        'mlm_probability': 1., 'embed_dim': 4, 'vision_width': 768,
        'bert_config': 'synthetic', 'image_res': 2, 'queue_size': 4, 'momentum': .9,
        'view_fusion_gate': {'hidden_dim': 8},
        'view_fusion_residual': {'hidden_dim': 8, 'initial_gate': .1},
        'view_fusion_transformer': {'num_heads': 2, 'num_layers': 2, 'mlp_ratio': .02, 'dropout': 0.},
        'view_fusion_cross_attention': {'num_heads': 2, 'num_layers': 2, 'mlp_ratio': .02, 'dropout': 0.,
                                      'anchor_view': 'original', 'memory_views': ['lung', 'heart']},
    }


TOKENIZER = SimpleNamespace(pad_token_id=0, cls_token_id=1, mask_token_id=2)
