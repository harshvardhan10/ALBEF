"""cross_attention fusion, extracted unchanged from the thesis baseline."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AnchoredCrossAttentionLayer(nn.Module):
    """One post-norm Transformer-style cross-attention layer.

    Query shape:  [1, B*N, D]  (original-view anchor token)
    Memory shape: [2, B*N, D]  (lung and heart tokens)

    Only the query stream is updated. The lung/heart memory remains fixed
    within a forward pass and is reused by the second cross-attention layer.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        mlp_ratio = float(mlp_ratio)
        dropout = float(dropout)

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        feedforward_dim = int(round(hidden_dim * mlp_ratio))

        # Sequence-first layout is retained for compatibility with the older
        # PyTorch version used by the ALBEF environment.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.linear1 = nn.Linear(hidden_dim, feedforward_dim)
        self.activation = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 3 or memory.ndim != 3:
            raise ValueError(
                "Expected sequence-first query/memory tensors with rank 3; "
                f"got query={tuple(query.shape)}, memory={tuple(memory.shape)}"
            )
        if query.shape[0] != 1:
            raise ValueError(
                f"Anchored query sequence length must be 1; got {query.shape[0]}"
            )
        if memory.shape[0] != 2:
            raise ValueError(
                f"Auxiliary memory sequence length must be 2; got {memory.shape[0]}"
            )
        if query.shape[1:] != memory.shape[1:]:
            raise ValueError(
                "Query and memory must share batch-group and hidden dimensions; "
                f"got query={tuple(query.shape)}, memory={tuple(memory.shape)}"
            )

        # For each head and each aligned token position, attention logits have
        # shape [1, 2]: the original query chooses between lung and heart.
        attn_output, _ = self.cross_attn(
            query=query,
            key=memory,
            value=memory,
            need_weights=False,
        )
        query = self.norm1(query + self.attn_dropout(attn_output))

        ffn_output = self.linear2(
            self.ffn_dropout(
                self.activation(self.linear1(query))
            )
        )
        query = self.norm2(query + self.output_dropout(ffn_output))
        return query


class ViewCrossAttentionFusion(nn.Module):
    """Fuse aligned views by anchoring on the original-view token.

    Inputs all have shape [B, N, D]. At every token position p independently:

        query  = original_p
        memory = [lung_p, heart_p]

    The query is updated by two stacked cross-attention layers. The final query
    becomes the fused token, yielding [B, N, D] with no view mean pooling.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_dim = int(hidden_dim)
        num_heads = int(num_heads)
        num_layers = int(num_layers)
        mlp_ratio = float(mlp_ratio)
        dropout = float(dropout)

        if num_layers != 2:
            raise ValueError(
                "This ablation is defined as a 2-layer anchored cross-attention "
                f"Transformer; got num_layers={num_layers}"
            )
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.hidden_dim = hidden_dim
        self.num_views = 3

        # Original/lung/heart identities. These are added once before the first
        # cross-attention layer. Shape broadcasts over the B*N token groups.
        self.view_embeddings = nn.Parameter(
            torch.zeros(self.num_views, 1, hidden_dim)
        )
        nn.init.normal_(self.view_embeddings, mean=0.0, std=0.02)

        self.layers = nn.ModuleList(
            [
                AnchoredCrossAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        original_tokens: torch.Tensor,
        lung_tokens: torch.Tensor,
        heart_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if not (
            original_tokens.shape == lung_tokens.shape == heart_tokens.shape
        ):
            raise ValueError(
                "The three view token tensors must have identical shapes; got "
                f"original={tuple(original_tokens.shape)}, "
                f"lung={tuple(lung_tokens.shape)}, "
                f"heart={tuple(heart_tokens.shape)}"
            )
        if original_tokens.ndim != 3:
            raise ValueError(
                "Expected view tokens with shape [B,N,D]; got "
                f"{tuple(original_tokens.shape)}"
            )

        batch_size, num_tokens, hidden_dim = original_tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dimension {self.hidden_dim}; got {hidden_dim}"
            )

        # Flatten B and N so every aligned token position is an independent
        # cross-attention problem. Sequence length is 1 for the anchor query.
        query = original_tokens.reshape(
            1,
            batch_size * num_tokens,
            hidden_dim,
        )
        query = query + self.view_embeddings[0:1]

        # Lung and heart are a length-2 auxiliary memory for every token group.
        # [B,N,D] x 2 -> [2,B,N,D] -> [2,B*N,D]
        memory_tokens = (lung_tokens, heart_tokens)
        memory = torch.stack(memory_tokens, dim=0).reshape(
            2,
            batch_size * num_tokens,
            hidden_dim,
        )
        memory = memory + self.view_embeddings[1:3]

        # Layer 1: O_p queries [L_p,H_p] -> O'_p
        # Layer 2: O'_p queries the same [L_p,H_p] memory -> O''_p
        for layer in self.layers:
            query = layer(query, memory)

        # No mean pooling: the twice-updated anchor query is the fused token.
        return query.squeeze(0).reshape(batch_size, num_tokens, hidden_dim)
