# Independent multiview references

multiview_models_6320a2e.json.gz contains the six original model sources at ALBEF commit 6320a2e93ce63bf2a77f3c716278c4379e9114f1. These are test-only references, not runtime implementations. They preserve the original ALBEF licensing. Tests compile these sources with small synthetic vision/text dependencies to check orchestration, loss/gradient/state behavior and fusion equivalence without pretending to reproduce the real encoders.

SHA-256 source hashes:

- models/model_pretrain_multiview_cross_attention_fusion.py: `12109d66cb288a2933a283bfe5956d58440e03a636806a1992ee8340e146e60c`
- models/model_pretrain_multiview_fusion.py: `67418d41f7a5a8cbc23b3a01feed5584024657622743f29a20c76555ec7240b6`
- models/model_pretrain_multiview_gated_fusion.py: `af2f8f4a38cb9fefb04d24582fabf58255c54be0318bc53d79ad9ed572be1aa8`
- models/model_pretrain_multiview_mean_fusion.py: `3fe3aa033e32618b64a042fa8346799488055c9fe81f4e0c48c3058f27df0071`
- models/model_pretrain_multiview_residual_fusion.py: `82da47f5a047db03030ee405bf0b07381ac3ce7735ef79d0562a63ea319e72b4`
- models/model_pretrain_multiview_transformer_fusion.py: `444158f6b0b041b8f9b6703dd0b64133059eebaa321a307b77ecd0867d819c6c`
