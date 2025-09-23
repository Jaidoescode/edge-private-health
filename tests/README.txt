# Edge-Private Health — Cough Detection (ESC-50)

[![CI](https://github.com/jaidoescode/edge-private-health/actions/workflows/ci.yml/badge.svg)](https://github.com/jaidoescode/edge-private-health/actions/workflows/ci.yml)

**What this is:** a small, production-minded ML project that detects **cough vs. not-cough** from audio, with:
- End-to-end pipeline: data prep → log-mel features → TinyCNN training → eval → artifacts  
- Federated learning demo (FedAvg) to simulate privacy-aware training  
- ONNX export + CPU benchmark for deployability  
- Tests + CI so the repo stays healthy and reproducible  

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


📊 Results
Model Evaluation
Model	Accuracy	F1	AUC
Baseline	0.67	0.60	1.00
FedAvg	0.25	0.40	0.74
ONNX Inference Latency (CPU, 100 runs)
Metric	Time (ms)
Mean	0.71
P50	0.66
P90	0.79
P99	1.76
Input shape: [1, 1, 64, 201]