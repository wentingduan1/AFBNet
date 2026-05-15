"""
Smoke and shape tests for AFBNet and submodules.
Run from project root: python test.py
Verbose: python test.py -v
"""
from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import BertConfig, BertModel  # noqa: E402

from afbnet import AFBNet, AFBNetConfig  # noqa: E402
from afbnet.adaptive_fusion import AdaptiveFusion  # noqa: E402
from afbnet.bert_fusion import MultiLayerBertFusion  # noqa: E402
from afbnet.mask_knowledge import MaskKnowledgeEmbedding  # noqa: E402


class TestMultiLayerBertFusion(unittest.TestCase):
    def test_output_shape(self) -> None:
        b, l, h, n = 2, 7, 48, 3
        layers = [torch.randn(b, l, h) for _ in range(n)]
        m = MultiLayerBertFusion(hidden_size=h, num_layers=n, dropout=0.0)
        out = m(layers)
        self.assertEqual(out.shape, (b, l, h))
        self.assertFalse(torch.isnan(out).any())


class TestMaskKnowledgeEmbedding(unittest.TestCase):
    def test_output_shape(self) -> None:
        b, l, e, h = 2, 9, 512, 64
        tok = torch.randn(b, l, e)
        bert = torch.randn(b, l, h)
        m = MaskKnowledgeEmbedding(bert_dim=h, embed_dim=e, mask_threshold=0.03)
        out = m(tok, bert)
        self.assertEqual(out.shape, (b, l, e))


class TestAdaptiveFusion(unittest.TestCase):
    def test_output_shape(self) -> None:
        b, l, d_model, d_bert = 2, 11, 512, 64
        q = torch.randn(b, l, d_model)
        bert = torch.randn(b, l, d_bert)
        m = AdaptiveFusion(d_model=d_model, d_bert=d_bert, attn_dim=d_model, dropout=0.0)
        out = m(q, bert)
        self.assertEqual(out.shape, (b, l, d_model))


class TestAFBNetTiny(unittest.TestCase):
    """Full model with a randomly initialized tiny BERT (no checkpoint download)."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.vocab = 512
        bert_cfg = BertConfig(
            vocab_size=self.vocab,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=128,
        )
        self.bert = BertModel(bert_cfg)
        self.net_cfg = AFBNetConfig(
            vocab_size=self.vocab,
            bert_name="local-tiny-bert",
            pad_id=0,
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.0,
            mask_theta=0.03,
            max_len=64,
            freeze_bert=True,
        )
        self.model = AFBNet(self.net_cfg, bert=self.bert)

    def test_encode_shape(self) -> None:
        b, ls = 2, 16
        src = torch.randint(1, self.vocab, (b, ls))
        mem = self.model.encode(src, src_key_padding_mask=None)
        self.assertEqual(mem.shape, (b, ls, self.net_cfg.d_model))

    def test_forward_logits_shape(self) -> None:
        b, ls, lt = 2, 14, 12
        src = torch.randint(1, self.vocab, (b, ls))
        tgt = torch.randint(1, self.vocab, (b, lt))
        logits = self.model(src, tgt)
        self.assertEqual(logits.shape, (b, lt, self.vocab))

    def test_forward_backward_no_nan(self) -> None:
        b, ls, lt = 2, 12, 10
        src = torch.randint(1, self.vocab, (b, ls))
        tgt = torch.randint(1, self.vocab, (b, lt))
        dec_in = tgt[:, :-1]
        dec_out = tgt[:, 1:]
        pad = 0
        tgt_pad = dec_in == pad
        logits = self.model(src, dec_in, tgt_key_padding_mask=tgt_pad)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, self.vocab),
            dec_out.reshape(-1),
            ignore_index=pad,
        )
        loss.backward()
        self.assertFalse(torch.isnan(loss).item())
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any(), msg=f"nan grad in {name}")


class TestSyntheticTSV(unittest.TestCase):
    """Optional: bundled full_de_en train.tsv is readable."""

    def test_train_tsv_header_and_row(self) -> None:
        path = ROOT / "data" / "full_de_en" / "train.tsv"
        if not path.is_file():
            self.skipTest(f"Missing {path}")
        with path.open(encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            self.assertEqual(r.fieldnames, ["source", "target"])
            row = next(iter(r))
            self.assertIn("source", row)
            self.assertIn("target", row)
            self.assertTrue(len(row["source"]) > 0)
            self.assertTrue(len(row["target"]) > 0)


if __name__ == "__main__":
    unittest.main()
