"""
Adaptive fusion at encoder output (Eq. 7–9):
W_AB = softmax(Q K^T / sqrt(d)) V ; η = σ(X Q + Y W_AB) ; W_f = (1-η) Q + η W_AB
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class AdaptiveFusion(nn.Module):
    def __init__(self, d_model: int = 512, d_bert: int = 768, attn_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.attn_dim = attn_dim
        self.scale = math.sqrt(attn_dim)
        self.q_proj = nn.Linear(d_model, attn_dim, bias=False)
        self.k_proj = nn.Linear(d_bert, attn_dim, bias=False)
        self.v_proj = nn.Linear(d_bert, d_model, bias=False)
        self.x_proj = nn.Linear(d_model, d_model, bias=True)
        self.y_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_out: torch.Tensor, bert_fused: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(encoder_out)
        k = self.k_proj(bert_fused)
        v = self.v_proj(bert_fused)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        w_ab = torch.matmul(attn, v)
        eta = torch.sigmoid(self.x_proj(encoder_out) + self.y_proj(w_ab))
        return (1.0 - eta) * encoder_out + eta * w_ab
