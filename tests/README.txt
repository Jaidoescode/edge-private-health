# Edge-Private Health — Cough Detection (ESC-50)

[![CI](https://github.com/jaidoescode/edge-private-health/actions/workflows/ci.yml/badge.svg)](https://github.com/jaidoescode/edge-private-health/actions/workflows/ci.yml)

**What this is:** a small, production-minded ML project that detects **cough vs. not-cough** from audio, with:
- **End-to-end pipeline:** data prep → log-mel features → TinyCNN training → eval → artifacts
- **Federated learning demo** (FedAvg) to simulate privacy-aware training
- **ONNX export + CPU benchmark** for deployability
- **Tests + CI** so the repo stays healthy and reproducible

> ⚠️ Not for medical use. Demo only.

---

## Quickstart

```bash
# 0) (optional) create a venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Prepare ESC-50 subset locally (downloads ~600MB once)
python -m src.data.prepare --root ./data --seed 42

# 2) Train centralized baseline
python -m src.train.train_baseline --data_root ./data --epochs 10

# 3) (Optional) Federated simulation (FedAvg)
python -m src.federated.fed_sim --data_root ./data --clients 5 --rounds 3 --local_epochs 1

# 4) Evaluate checkpoints
python -m src.eval.report --data_root ./data --ckpt ./artifacts/baseline.pt
python -m src.eval.report --data_root ./data --ckpt ./artifacts/fedavg.pt

Add tests later (unit tests for data shapes, metrics).