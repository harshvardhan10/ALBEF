# Full model implementations

These `.py` files contain the actual model bodies. For experiment-to-model navigation, use the [model-code index](../../../docs/model_code.md).

- `model_pretrain.py`: shared baseline ALBEF for A0–A4 and single-view input experiments.
- `model_pretrain_multiview_fusion.py`: FC token fusion.
- `model_pretrain_multiview_mean_fusion.py`: mean token fusion.
- `model_pretrain_multiview_gated_fusion.py`: learned study-level softmax gates.
- `model_pretrain_multiview_residual_fusion.py`: original-anchored residual fusion.
- `model_pretrain_multiview_transformer_fusion.py`: two-layer cross-view self-attention.
- `model_pretrain_multiview_cross_attention_fusion.py`: two-layer original-query cross-attention.
- `vit.py`, `xbert.py`, `tokenization_bert.py`: shared encoders/tokenizer.

The guidance penalties are implemented in the anatomy trainers/helpers, not separate model subclasses. The other model files preserve inherited NLVR, retrieval, VQA and visual-entailment functionality. No architecture file was merged or rewritten during navigation work.
