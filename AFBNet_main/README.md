# AFBNet

This repository implements **AFBNet (Adaptive Fusion BERT Network)** from the paper 


An Intelligent Assessment Method for English Translation Teaching Model Based on Improved BERT and Attention Mechanism

---

## Requirements

- Python ≥ 3.10  
- PyTorch and Transformers (see `requirements.txt`)

```bash
pip install -r requirements.txt
```

---

## Dataset


| File | Default size (sentence pairs, excluding header) |
|------|--------------------------------------------------|
| `data/full_de_en/train.tsv` | 120,000 |
| `data/full_de_en/dev.tsv`   | 5,000 |
| `data/full_de_en/test.tsv`  | 5,000 |
| `data/full_de_en/dataset_manifest.json` | Line counts and SHA256 |



## Training

By default, `train.py` reads `data/full_de_en/` (same layout as the full preset):

```bash
python train.py
```

## Test

```bash
python test.py
```