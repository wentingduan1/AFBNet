"""
Mask knowledge matrix + gated embeddings (Eq. 3–6).
S from similarity of token embeddings with projected [CLS]; threshold mask; G = λ B_e + (1-λ) W'.
B_e is B projected to embedding dim (paper uses B at token level; dimensions matched via linear maps).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class MaskKnowledgeEmbedding(nn.Module):
    def __init__(
        self,
        bert_dim: int = 768,
        embed_dim: int = 512,
        mask_threshold: float = 0.03,
        mask_fill: float = -1e-9,
        scale_c: float = 512.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_threshold = mask_threshold
        self.mask_fill = mask_fill
        self.scale_c = scale_c
        self.cls_proj = nn.Linear(bert_dim, embed_dim, bias=True)
        self.b_to_embed = nn.Linear(bert_dim, embed_dim, bias=False)
        self.b_gate = nn.Linear(bert_dim, embed_dim, bias=True)
        self.w_gate = nn.Linear(embed_dim, embed_dim, bias=True)
        self.lambda_proj = nn.Linear(2 * embed_dim, embed_dim, bias=True)
        self.lambda_to_scalar = nn.Linear(embed_dim, 1, bias=True)

    def forward(self, token_embeds: torch.Tensor, bert_fused: torch.Tensor) -> torch.Tensor:
        """
        token_embeds W: (B, L, E)
        bert_fused B: (B, L, H_bert)
        returns gated embeddings (B, L, E)
        """
        cls = bert_fused[:, 0, :]
        h = self.cls_proj(cls)
        logits = (token_embeds * h.unsqueeze(1)).sum(dim=-1)
        logits = logits / self.scale_c
        s = torch.softmax(logits, dim=-1)
        m = (s < self.mask_threshold).float().unsqueeze(-1)
        w_masked = token_embeds.masked_fill(m.bool(), self.mask_fill)
        b_e = self.b_to_embed(bert_fused)
        b_p = self.b_gate(bert_fused)
        w_p = self.w_gate(w_masked)
        lam_h = torch.relu(self.lambda_proj(torch.cat([b_p, w_p], dim=-1)))
        # Sigmoid yields a stable convex combination; ReLU-only λ is unbounded in Eq.6 of the paper text.
        lam = torch.sigmoid(self.lambda_to_scalar(lam_h))
        return lam * b_e + (1.0 - lam) * w_masked
