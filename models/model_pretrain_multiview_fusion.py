"""
Multi-view ALBEF pretraining model.

Three view-specific ViT encoders process:
    1. original CXR
    2. lung-masked CXR
    3. heart-masked CXR

Corresponding ViT tokens are concatenated along the hidden dimension and fused
with one learned fully connected layer:

    [B, N, 768] x 3 -> [B, N, 2304] -> Linear(2304, 768) -> LayerNorm

The resulting fused token sequence is the *only* visual representation used by
the standard ALBEF objectives:
    - ITC / ITA: fused CLS -> vision_proj -> contrastive loss
    - ITM: text cross-attends to all fused image tokens
    - MLM: masked text cross-attends to all fused image tokens

The momentum pathway mirrors all three ViTs and the fusion layer so the ALBEF
momentum targets and queues remain consistent with the fused online model.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM


class ALBEF(nn.Module):
    def __init__(
        self,
        text_encoder=None,
        tokenizer=None,
        config=None,
        temp=0.07,
        init_deit=True,
    ):
        super().__init__()

        if config is None:
            raise ValueError("config is required")

        self.tokenizer = tokenizer
        self.mlm_probability = float(config["mlm_probability"])
        embed_dim = int(config["embed_dim"])
        vision_width = int(config["vision_width"])

        if vision_width != 768:
            raise ValueError(
                "This implementation uses ViT-B/16 with hidden width 768; "
                f"got vision_width={vision_width}"
            )

        # ------------------------------------------------------------------
        # Online visual branches
        # ------------------------------------------------------------------
        self.visual_encoder_original = self._build_visual_encoder(config)
        self.visual_encoder_lung = self._build_visual_encoder(config)
        self.visual_encoder_heart = self._build_visual_encoder(config)

        if init_deit:
            self._initialize_all_visual_encoders_from_deit()

        # Learned token-wise fusion.
        # Every token position p is fused independently:
        #   cat(z_orig[p], z_lung[p], z_heart[p]) -> 768
        self.view_fusion = nn.Linear(3 * vision_width, vision_width)
        self.view_fusion_norm = nn.LayerNorm(vision_width)
        self._initialize_fusion_equal_average()

        # ------------------------------------------------------------------
        # Shared ALBEF text / projection / ITM components
        # ------------------------------------------------------------------
        bert_config = BertConfig.from_json_file(config["bert_config"])
        self.text_encoder = BertForMaskedLM.from_pretrained(
            text_encoder,
            config=bert_config,
        )

        text_width = int(self.text_encoder.config.hidden_size)
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * float(config.get("temp", temp)))
        self.queue_size = int(config["queue_size"])
        self.momentum = float(config["momentum"])
        self.itm_head = nn.Linear(text_width, 2)

        # ------------------------------------------------------------------
        # Momentum pathway
        # ------------------------------------------------------------------
        self.visual_encoder_original_m = self._build_visual_encoder(config)
        self.visual_encoder_lung_m = self._build_visual_encoder(config)
        self.visual_encoder_heart_m = self._build_visual_encoder(config)

        self.view_fusion_m = nn.Linear(3 * vision_width, vision_width)
        self.view_fusion_norm_m = nn.LayerNorm(vision_width)

        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(
            text_encoder,
            config=bert_config,
        )
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [
            [self.visual_encoder_original, self.visual_encoder_original_m],
            [self.visual_encoder_lung, self.visual_encoder_lung_m],
            [self.visual_encoder_heart, self.visual_encoder_heart_m],
            [self.view_fusion, self.view_fusion_m],
            [self.view_fusion_norm, self.view_fusion_norm_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]

        # Momentum modules begin as exact copies and never receive gradients.
        self.copy_params()

        # ------------------------------------------------------------------
        # One queue for the one fused study representation.
        # Do not maintain separate original/lung/heart queues.
        # ------------------------------------------------------------------
        self.register_buffer(
            "image_queue",
            torch.randn(embed_dim, self.queue_size),
        )
        self.register_buffer(
            "text_queue",
            torch.randn(embed_dim, self.queue_size),
        )
        self.register_buffer(
            "queue_ptr",
            torch.zeros(1, dtype=torch.long),
        )

        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.text_queue = F.normalize(self.text_queue, dim=0)

    @staticmethod
    def _build_visual_encoder(config):
        return VisionTransformer(
            img_size=int(config["image_res"]),
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

    def _initialize_all_visual_encoders_from_deit(self) -> None:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=(
                "https://dl.fbaipublicfiles.com/deit/"
                "deit_base_patch16_224-b5f2ef4d.pth"
            ),
            map_location="cpu",
            check_hash=True,
        )
        base_state = checkpoint["model"]

        for name, encoder in (
            ("original", self.visual_encoder_original),
            ("lung", self.visual_encoder_lung),
            ("heart", self.visual_encoder_heart),
        ):
            state = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in base_state.items()
            }
            state["pos_embed"] = interpolate_pos_embed(
                state["pos_embed"],
                encoder,
            )
            msg = encoder.load_state_dict(state, strict=False)
            print(f"[DeiT init] {name}: {msg}", flush=True)

    @torch.no_grad()
    def _initialize_fusion_equal_average(self) -> None:
        """Initialize FC fusion as an equal average of the three token streams.

        For output dimension d:
            y[d] = (orig[d] + lung[d] + heart[d]) / 3

        This gives the new layer a stable, non-random starting representation
        while leaving every FC weight learnable.
        """
        out_dim = self.view_fusion.out_features
        if self.view_fusion.in_features != 3 * out_dim:
            raise ValueError("Unexpected fusion dimensions")

        self.view_fusion.weight.zero_()
        self.view_fusion.bias.zero_()
        eye = torch.eye(out_dim, dtype=self.view_fusion.weight.dtype)
        self.view_fusion.weight[:, 0:out_dim].copy_(eye / 3.0)
        self.view_fusion.weight[:, out_dim:2 * out_dim].copy_(eye / 3.0)
        self.view_fusion.weight[:, 2 * out_dim:3 * out_dim].copy_(eye / 3.0)

        self.view_fusion_norm.weight.fill_(1.0)
        self.view_fusion_norm.bias.zero_()

    def _fuse_tokens(
        self,
        original_tokens: torch.Tensor,
        lung_tokens: torch.Tensor,
        heart_tokens: torch.Tensor,
        *,
        momentum: bool,
    ) -> torch.Tensor:
        if not (
            original_tokens.shape == lung_tokens.shape == heart_tokens.shape
        ):
            raise ValueError(
                "The three ViT token tensors must have identical shapes; got "
                f"original={tuple(original_tokens.shape)}, "
                f"lung={tuple(lung_tokens.shape)}, "
                f"heart={tuple(heart_tokens.shape)}"
            )

        concatenated = torch.cat(
            [original_tokens, lung_tokens, heart_tokens],
            dim=-1,
        )

        if momentum:
            return self.view_fusion_norm_m(
                self.view_fusion_m(concatenated)
            )

        return self.view_fusion_norm(
            self.view_fusion(concatenated)
        )

    def encode_image_views(
        self,
        image_original: torch.Tensor,
        image_lung: torch.Tensor,
        image_heart: torch.Tensor,
        *,
        momentum: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return fused image tokens and normalized ITC image features.

        Returns:
            image_embeds: [B, num_tokens, 768]
            image_feat:   [B, embed_dim]
        """
        if not (
            image_original.shape == image_lung.shape == image_heart.shape
        ):
            raise ValueError(
                "The three input image batches must have identical shapes; got "
                f"original={tuple(image_original.shape)}, "
                f"lung={tuple(image_lung.shape)}, "
                f"heart={tuple(image_heart.shape)}"
            )

        if momentum:
            z_original = self.visual_encoder_original_m(image_original)
            z_lung = self.visual_encoder_lung_m(image_lung)
            z_heart = self.visual_encoder_heart_m(image_heart)
            image_embeds = self._fuse_tokens(
                z_original,
                z_lung,
                z_heart,
                momentum=True,
            )
            image_feat = F.normalize(
                self.vision_proj_m(image_embeds[:, 0, :]),
                dim=-1,
            )
        else:
            z_original = self.visual_encoder_original(image_original)
            z_lung = self.visual_encoder_lung(image_lung)
            z_heart = self.visual_encoder_heart(image_heart)
            image_embeds = self._fuse_tokens(
                z_original,
                z_lung,
                z_heart,
                momentum=False,
            )
            image_feat = F.normalize(
                self.vision_proj(image_embeds[:, 0, :]),
                dim=-1,
            )

        return image_embeds, image_feat

    def get_image_features(
        self,
        image_original: torch.Tensor,
        image_lung: torch.Tensor,
        image_heart: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience method for zero-shot classification/validation."""
        _, image_feat = self.encode_image_views(
            image_original,
            image_lung,
            image_heart,
            momentum=False,
        )
        return image_feat

    def forward(
        self,
        image_original,
        image_lung,
        image_heart,
        text,
        alpha=0,
    ):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        # ================================================================
        # Online fused visual representation
        # ================================================================
        image_embeds, image_feat = self.encode_image_views(
            image_original,
            image_lung,
            image_heart,
            momentum=False,
        )
        image_atts = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=image_original.device,
        )

        # Text-only representation.
        text_output = self.text_encoder.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(
            self.text_proj(text_embeds[:, 0, :]),
            dim=-1,
        )

        # ================================================================
        # ITC / ITA with fused momentum image features
        # ================================================================
        with torch.no_grad():
            self._momentum_update()

            image_embeds_m, image_feat_m = self.encode_image_views(
                image_original,
                image_lung,
                image_heart,
                momentum=True,
            )
            image_feat_all = torch.cat(
                [
                    image_feat_m.t(),
                    self.image_queue.clone().detach(),
                ],
                dim=1,
            )

            text_output_m = self.text_encoder_m.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_feat_m = F.normalize(
                self.text_proj_m(
                    text_output_m.last_hidden_state[:, 0, :]
                ),
                dim=-1,
            )
            text_feat_all = torch.cat(
                [
                    text_feat_m.t(),
                    self.text_queue.clone().detach(),
                ],
                dim=1,
            )

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros_like(sim_i2t_m)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = (
                alpha * F.softmax(sim_i2t_m, dim=1)
                + (1.0 - alpha) * sim_targets
            )
            sim_t2i_targets = (
                alpha * F.softmax(sim_t2i_m, dim=1)
                + (1.0 - alpha) * sim_targets
            )

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets,
            dim=1,
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets,
            dim=1,
        ).mean()
        loss_ita = (loss_i2t + loss_t2i) / 2.0

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        # ================================================================
        # ITM: positive pair
        # Text cross-attends to the full fused visual token sequence.
        # ================================================================
        output_pos = self.text_encoder.bert(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode="fusion",
        )

        bs = image_original.size(0)
        if bs < 2:
            raise ValueError(
                "ALBEF ITM hard-negative sampling requires batch size >= 2. "
                "Use drop_last=True or increase batch size."
            )

        with torch.no_grad():
            # image -> candidate texts
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            # text -> candidate images
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

            neg_image_indices = self._sample_negative_indices(weights_t2i)
            neg_text_indices = self._sample_negative_indices(weights_i2t)

        # One negative fused image per text.
        image_embeds_neg = image_embeds[neg_image_indices]

        # One negative text per fused image.
        text_embeds_neg = text_embeds[neg_text_indices]
        text_atts_neg = text.attention_mask[neg_text_indices]

        text_embeds_all = torch.cat(
            [text_embeds, text_embeds_neg],
            dim=0,
        )
        text_atts_all = torch.cat(
            [text.attention_mask, text_atts_neg],
            dim=0,
        )

        image_embeds_all = torch.cat(
            [image_embeds_neg, image_embeds],
            dim=0,
        )
        image_atts_all = torch.cat(
            [image_atts, image_atts],
            dim=0,
        )

        output_neg = self.text_encoder.bert(
            encoder_embeds=text_embeds_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
            mode="fusion",
        )

        vl_embeddings = torch.cat(
            [
                output_pos.last_hidden_state[:, 0, :],
                output_neg.last_hidden_state[:, 0, :],
            ],
            dim=0,
        )
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat(
            [
                torch.ones(bs, dtype=torch.long),
                torch.zeros(2 * bs, dtype=torch.long),
            ],
            dim=0,
        ).to(image_original.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        # ================================================================
        # MLM: masked report cross-attends to fused visual tokens.
        # Momentum teacher also conditions on fused momentum visual tokens.
        # ================================================================
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(
            labels.shape,
            self.mlm_probability,
            device=labels.device,
        )
        input_ids, labels = self.mask(
            input_ids,
            self.text_encoder.config.vocab_size,
            image_original.device,
            targets=labels,
            probability_matrix=probability_matrix,
        )

        with torch.no_grad():
            logits_m = self.text_encoder_m(
                input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds_m,
                encoder_attention_mask=image_atts,
                return_dict=True,
                return_logits=True,
            )

        mlm_output = self.text_encoder(
            input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            labels=labels,
            soft_labels=F.softmax(logits_m, dim=-1),
            alpha=alpha,
        )
        loss_mlm = mlm_output.loss

        return loss_mlm, loss_ita, loss_itm

    @staticmethod
    def _sample_negative_indices(weights: torch.Tensor) -> torch.Tensor:
        """Sample one non-self in-batch negative for every row.

        `weights` is expected to have its diagonal zeroed already.
        If numerical underflow leaves a row with zero probability mass, fall
        back to a uniform distribution over all *other* batch elements.
        """
        if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
            raise ValueError(
                "Expected square in-batch negative-sampling weights; got "
                f"{tuple(weights.shape)}"
            )

        bs = weights.shape[0]
        if bs < 2:
            raise ValueError("Need at least two samples for a non-self negative")

        sampled = []
        for b in range(bs):
            probs = weights[b].clone()
            probs[b] = 0

            if (
                not torch.isfinite(probs).all()
                or float(probs.sum()) <= 0.0
            ):
                probs = torch.ones_like(probs)
                probs[b] = 0

            probs = probs / probs.sum()
            sampled.append(torch.multinomial(probs, 1))

        return torch.cat(sampled, dim=0)

    @torch.no_grad()
    def copy_params(self):
        """Copy online parameters to momentum modules and freeze momentum."""
        for model_pair in self.model_pairs:
            online, momentum = model_pair
            online_params = list(online.parameters())
            momentum_params = list(momentum.parameters())

            if len(online_params) != len(momentum_params):
                raise RuntimeError(
                    "Online/momentum module parameter counts differ"
                )

            for param, param_m in zip(online_params, momentum_params):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def reset_momentum_from_online(self):
        """Use after assembling the model from three separate checkpoints."""
        self.copy_params()

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(),
                model_pair[1].parameters(),
            ):
                param_m.data.mul_(self.momentum).add_(
                    param.data,
                    alpha=1.0 - self.momentum,
                )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        """Ring-buffer enqueue that also works when queue_size % batch != 0."""
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        if image_feats.shape != text_feats.shape:
            raise ValueError(
                "Gathered image/text feature shapes differ: "
                f"{tuple(image_feats.shape)} vs {tuple(text_feats.shape)}"
            )

        batch_size = int(image_feats.shape[0])
        if batch_size > self.queue_size:
            # Retain only the most recent queue_size samples.
            image_feats = image_feats[-self.queue_size :]
            text_feats = text_feats[-self.queue_size :]
            batch_size = self.queue_size

        ptr = int(self.queue_ptr.item())
        first = min(batch_size, self.queue_size - ptr)

        self.image_queue[:, ptr : ptr + first] = image_feats[:first].T
        self.text_queue[:, ptr : ptr + first] = text_feats[:first].T

        remaining = batch_size - first
        if remaining > 0:
            self.image_queue[:, :remaining] = image_feats[first:].T
            self.text_queue[:, :remaining] = text_feats[first:].T

        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def mask(
        self,
        input_ids,
        vocab_size,
        device,
        targets=None,
        masked_indices=None,
        probability_matrix=None,
    ):
        if masked_indices is None:
            if probability_matrix is None:
                raise ValueError(
                    "probability_matrix is required when masked_indices is None"
                )
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100

        # 80% -> [MASK]
        indices_replaced = (
            torch.bernoulli(
                torch.full(
                    input_ids.shape,
                    0.8,
                    device=input_ids.device,
                )
            ).bool()
            & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% -> random word (half of the remaining 20%)
        indices_random = (
            torch.bernoulli(
                torch.full(
                    input_ids.shape,
                    0.5,
                    device=input_ids.device,
                )
            ).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            vocab_size,
            input_ids.shape,
            dtype=torch.long,
            device=device,
        )
        input_ids[indices_random] = random_words[indices_random]

        # Remaining 10% stay unchanged.
        if targets is not None:
            return input_ids, targets
        return input_ids


@torch.no_grad()
def concat_all_gather(tensor):
    """All-gather without gradients; identity in non-distributed mode."""
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        return tensor

    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return tensor

    tensors_gather = [
        torch.empty_like(tensor)
        for _ in range(world_size)
    ]
    torch.distributed.all_gather(
        tensors_gather,
        tensor,
        async_op=False,
    )
    return torch.cat(tensors_gather, dim=0)
