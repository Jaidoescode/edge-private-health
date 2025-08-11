# Edge-Private Health (MVP) — Cough vs Other (ESC-50 subset)

**Goal for today:** ship a minimal, recruiter-ready repo that shows:
- clear problem framing,
- a working baseline (tiny CNN) on a public dataset,
- a toy **federated learning** simulation (FedAvg) over K clients,
- an optional **DP-SGD** mode (clip per-sample grads + Gaussian noise),
- clean structure, quickstart commands, and a short results report.

> This is an MVP meant to be finished **today**. You can iterate later (Raspberry Pi deploy, secure aggregation, model card).

## Quickstart

```bash
# 1) Create & activate a venv (recommended)
python3 -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Prepare data (downloads ESC-50 via torchaudio, filters cough vs other)
python -m src.data.prepare --root ./data --seed 42

# 4) Train a baseline model (single client)
python -m src.train.train_baseline --data_root ./data --epochs 3

# 5) Run a toy federated simulation (FedAvg with 5 clients)
python -m src.federated.fed_sim --data_root ./data --clients 5 --rounds 3 --local_epochs 1

# 6) Evaluate saved model
python -m src.eval.report --data_root ./data --ckpt ./artifacts/baseline.pt
```
## Reproducibility & Data

This repo **does not** track datasets or large artifacts. To reproduce results:

```bash
# 1) Prepare ESC-50 subset locally (downloads ~600MB once)
python -m src.data.prepare --root ./data --seed 42

# 2) Train centralized baseline
python -m src.train.train_baseline --data_root ./data --epochs 10

# 3) (Optional) Federated simulation
python -m src.federated.fed_sim --data_root ./data --clients 5 --rounds 3 --local_epochs 1

# 4) Evaluate
python -m src.eval.report --data_root ./data --ckpt ./artifacts/baseline.pt
python -m src.eval.report --data_root ./data --ckpt ./artifacts/fedavg.pt

## Results (MVP)
- Centralized: Acc <BASELINE_ACC>, F1 <BASELINE_F1>, AUC <BASELINE_AUC>
- FedAvg (5×3): Acc <FEDAVG_ACC>, F1 <FEDAVG_F1>, AUC <FEDAVG_AUC>

Artifacts: `./artifacts/baseline.pt`, `./artifacts/fedavg.pt`


## Why this is impressive enough today
- Uses a **public dataset (ESC-50)** and clean preprocessing.
- Implements **FedAvg** from scratch (simulated), not just centralized training.
- Adds an **optional DP-SGD** flag to showcase privacy awareness (replace with Opacus later for production-grade accounting).
- Clean repo + CI hooks you can expand.

## Repo map
```
edge-private-health/
├─ README.md
├─ requirements.txt
├─ pyproject.toml
├─ Makefile
├─ configs/
├─ src/
│  ├─ data/prepare.py
│  ├─ models/tiny_cnn.py
│  ├─ train/train_baseline.py
│  ├─ federated/fed_utils.py
│  ├─ federated/fed_sim.py
│  └─ eval/report.py
├─ docs/baseline.md
├─ scripts/
└─ tests/
```

## Roadmap (post-MVP)
- Quantize & export to TFLite/ONNX; measure on Raspberry Pi.
- Swap toy DP with **Opacus** and add ε accounting.
- Add **secure aggregation** simulation.
- Write a **Model Card** and an **Ablation report**.

## Results (MVP)
- Centralized: Acc 0.667, F1 0.600, AUC 1.000
- FedAvg (5×3): Acc 0.250, F1 0.400, AUC 0.741

Artifacts: `./artifacts/baseline.pt`, `./artifacts/fedavg.pt`
