"""
Multi-layer BERT fusion (Eq. 1–2 in the paper):
B_cat = Concat(B1, ..., Bn); B = GLU(B_cat @ L)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiLayerBertFusion(nn.Module):
    def __init__(self, hidden_size: int = 768, num_layers: int = 12, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        fused_in = hidden_size * num_layers
        # GLU: linear maps to 2 * hidden, then h * sigmoid(g)
        self.proj = nn.Linear(fused_in, 2 * hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, layer_hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
        """
        layer_hidden_states: list of (B, L, H) for layers 1..N (same as HF hidden_states[1:])
        returns B_fused: (B, L, H)
        """
        x = torch.cat(layer_hidden_states, dim=-1)
        x = self.dropout(x)
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)
