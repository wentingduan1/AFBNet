"""
AFBNet: Transformer seq2seq with
- multi-layer BERT fusion (frozen BERT),
- mask-knowledge gated embeddings,
- adaptive fusion on encoder outputs (paper: encoder-side fusion only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel

from .adaptive_fusion import AdaptiveFusion
from .bert_fusion import MultiLayerBertFusion
from .mask_knowledge import MaskKnowledgeEmbedding


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        x = x + self.pe[:, :t, :]
        return self.dropout(x)


@dataclass
class AFBNetConfig:
    vocab_size: int
    # Multilingual BERT matches synthetic DE->EN tokenization in the bundled demo.
    bert_name: str = "google-bert/bert-base-multilingual-cased"
    pad_id: int = 0
    d_model: int = 512
    nhead: int = 4
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.3
    mask_theta: float = 0.03
    max_len: int = 128
    freeze_bert: bool = True


class AFBNet(nn.Module):
    def __init__(self, cfg: AFBNetConfig, bert: PreTrainedModel | None = None):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.max_len, dropout=cfg.dropout)

        if bert is None:
            bert = AutoModel.from_pretrained(cfg.bert_name)
        self.bert = bert
        if cfg.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        bert_hidden = int(self.bert.config.hidden_size)
        n_bert_layers = int(self.bert.config.num_hidden_layers)
        self.bert_fusion = MultiLayerBertFusion(hidden_size=bert_hidden, num_layers=n_bert_layers, dropout=0.1)
        self.mask_embed = MaskKnowledgeEmbedding(
            bert_dim=bert_hidden,
            embed_dim=cfg.d_model,
            mask_threshold=cfg.mask_theta,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_encoder_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_decoder_layers)
        self.adaptive_fusion = AdaptiveFusion(
            d_model=cfg.d_model,
            d_bert=bert_hidden,
            attn_dim=cfg.d_model,
            dropout=0.2,
        )
        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def encode(self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        tok = self.embed(src_ids)
        if src_key_padding_mask is None:
            bert_attn = torch.ones(src_ids.shape[0], src_ids.shape[1], device=src_ids.device, dtype=torch.long)
        else:
            bert_attn = (~src_key_padding_mask).long()
        if self.cfg.freeze_bert:
            with torch.no_grad():
                hs = self.bert(
                    input_ids=src_ids,
                    attention_mask=bert_attn,
                    output_hidden_states=True,
                )
        else:
            hs = self.bert(
                input_ids=src_ids,
                attention_mask=bert_attn,
                output_hidden_states=True,
            )
        layers = hs.hidden_states[1:]
        b_fused = self.bert_fusion(layers)
        x = self.mask_embed(tok, b_fused)
        x = self.pos(x)
        mem = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        mem = self.adaptive_fusion(mem, b_fused)
        return mem

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mem = self.encode(src_ids, src_key_padding_mask)
        tgt_emb = self.pos(self.embed(tgt_ids))
        tlen = tgt_ids.size(1)
        try:
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(tlen, device=tgt_ids.device)
        except TypeError:
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(tlen).to(tgt_ids.device)
        dec = self.decoder(
            tgt_emb,
            mem,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_proj(dec)
