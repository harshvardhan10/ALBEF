"""Canonical identifiers used by the validated cross-attention extractor.

These are ViT self-attention Grad-CAM maps through anchored view fusion,
not direct fusion attention or text-image cross-attention maps.
"""
CROSS_ATTENTION_METHOD = 'albef_multiview_cross_attention_itc_margin_self_attention_gradcam'
CROSS_ATTENTION_FUSION = 'two_layer_original_anchored_cross_attention_lung_heart'
CROSS_ATTENTION_VERSION = '1.0-cross-attention-baseline'
COMPOSITE_CAM_DEFINITION = 'sum raw positive branch CAMs, normalize once'
ITC_MARGIN_TARGET = '(positive_similarity-negative_similarity)/temperature'
