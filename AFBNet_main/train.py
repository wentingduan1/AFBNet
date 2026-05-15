"""
Minimal training entrypoint for AFBNet on TSV parallel data (source \\t target).
Paper uses Fairseq + frozen BERT; this script is a compact PyTorch analogue for reproduction experiments.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from afbnet import AFBNet, AFBNetConfig  # noqa: E402


@dataclass
class TrainArgs:
    data_dir: Path = ROOT / "data" / "full_de_en"
    bert_name: str = "google-bert/bert-base-multilingual-cased"
    batch_size: int = 8
    max_len: int = 64
    lr: float = 1e-4
    weight_decay: float = 5e-4
    epochs: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_train_args() -> TrainArgs:
    p = argparse.ArgumentParser(description="Train AFBNet on parallel TSV (source<TAB>target).")
    p.add_argument(
        "--data_dir",
        type=Path,
        default=ROOT / "data" / "full_de_en",
        help="Directory containing train.tsv, dev.tsv, test.tsv",
    )
    p.add_argument("--bert_name", type=str, default=TrainArgs.bert_name)
    p.add_argument("--batch_size", type=int, default=TrainArgs.batch_size)
    p.add_argument("--max_len", type=int, default=TrainArgs.max_len)
    p.add_argument("--lr", type=float, default=TrainArgs.lr)
    p.add_argument("--weight_decay", type=float, default=TrainArgs.weight_decay)
    p.add_argument("--epochs", type=int, default=TrainArgs.epochs)
    p.add_argument("--device", type=str, default=None)
    ns = p.parse_args()
    device = ns.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return TrainArgs(
        data_dir=ns.data_dir,
        bert_name=ns.bert_name,
        batch_size=ns.batch_size,
        max_len=ns.max_len,
        lr=ns.lr,
        weight_decay=ns.weight_decay,
        epochs=ns.epochs,
        device=device,
    )


class ParallelTSV(Dataset):
    def __init__(self, path: Path):
        self.rows: list[tuple[str, str]] = []
        with path.open(encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                self.rows.append((row["source"], row["target"]))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.rows[idx]


def collate(batch: list[tuple[str, str]], tok: AutoTokenizer, max_len: int, pad_id: int):
    src_texts = [b[0] for b in batch]
    tgt_texts = [b[1] for b in batch]
    enc = tok(
        src_texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    dec = tok(
        tgt_texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    src_ids = enc["input_ids"]
    tgt_full = dec["input_ids"]
    src_pad = src_ids == pad_id
    tgt_pad = tgt_full == pad_id
    # Teacher forcing: decoder input is shifted labels; predict tgt_full[:, 1:]
    dec_in = tgt_full[:, :-1].contiguous()
    dec_out = tgt_full[:, 1:].contiguous()
    tgt_pad_in = dec_in == pad_id
    return src_ids, src_pad, dec_in, dec_out, tgt_pad_in


def train() -> None:
    args = parse_train_args()
    if not (args.data_dir / "train.tsv").exists():
        raise FileNotFoundError(
            f"Missing {args.data_dir / 'train.tsv'}. Run: python data/generate_synthetic_parallel.py --preset full"
        )

    tok = AutoTokenizer.from_pretrained(args.bert_name)
    pad_id = int(tok.pad_token_id or 0)
    cfg = AFBNetConfig(vocab_size=tok.vocab_size, bert_name=args.bert_name, pad_id=pad_id, max_len=args.max_len)
    bert = AutoModel.from_pretrained(args.bert_name)
    model = AFBNet(cfg, bert=bert).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    ds_tr = ParallelTSV(args.data_dir / "train.tsv")
    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok, args.max_len, pad_id),
    )

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        ntok = 0
        pbar = tqdm(dl_tr, desc=f"epoch {epoch+1}/{args.epochs}")
        for src_ids, src_pad, dec_in, dec_out, tgt_pad_in in pbar:
            src_ids = src_ids.to(args.device)
            src_pad = src_pad.to(args.device)
            dec_in = dec_in.to(args.device)
            dec_out = dec_out.to(args.device)
            tgt_pad_in = tgt_pad_in.to(args.device)

            logits = model(
                src_ids,
                dec_in,
                src_key_padding_mask=src_pad,
                tgt_key_padding_mask=tgt_pad_in,
                memory_key_padding_mask=src_pad,
            )
            ce = loss_fn(logits.reshape(-1, logits.size(-1)), dec_out.reshape(-1))
            opt.zero_grad(set_to_none=True)
            ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(ce.item())
            ntok += 1
            pbar.set_postfix(loss=f"{total/max(1,ntok):.4f}")

    out_path = ROOT / "afbnet_demo.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "bert_name": args.bert_name}, out_path)
    print(f"Saved checkpoint to {out_path}")


if __name__ == "__main__":
    train()
