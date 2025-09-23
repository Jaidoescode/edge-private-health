# Cough Detection with TinyCNN (ESC-50 Audio Dataset)

[![CI](https://github.com/jaidoescode/cough-detection-tinycnn/actions/workflows/ci.yml/badge.svg)](https://github.com/jaidoescode/cough-detection-tinycnn/actions/workflows/ci.yml)


**What this is:** A production-minded ML project that detects **cough vs. not-cough** from audio using a **TinyCNN** model, with:
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

# 1) Prepare ESC-50 subset locally (~600MB download once)
python -m src.data.prepare --root ./data --seed 42

# 2) Train centralized baseline
python -m src.train.train_baseline --data_root ./data --epochs 10

# 3) (optional) Run a federated learning simulation (FedAvg)
python -m src.federated.fed_sim --data_root ./data --clients 5 --rounds 3 --local_epochs 1

# 4) Evaluate checkpoints
python -m src.eval.report --data_root ./data --ckpt ./artifacts/baseline.pt
python -m src.eval.report --data_root ./data --ckpt ./artifacts/fedavg.pt

# file-level inference on a WAV
python -m src.infer --file data/raw/ESC-50-master/audio/1-19111-A-24.wav --ckpt ./artifacts/baseline.pt

# optional: adjust decision threshold (default 0.5)
python -m src.infer --file data/raw/ESC-50-master/audio/1-19111-A-24.wav --ckpt ./artifacts/baseline.pt --threshold 0.7

---

## Results(MVP)


| Model              | Accuracy |    F1 |   AUC | Notes                               |
| ------------------ | -------: | ----: | ----: | ----------------------------------- |
| Centralized        |    0.667 | 0.600 | 1.000 | Trained on full data centrally      |
| Federated (FedAvg) |    0.250 | 0.400 | 0.741 | 5 clients × 3 rounds, 1 local epoch |

## ONNX Export & CPU Benchmark

Export the trained model to **ONNX** and measure CPU latency.

```bash
python scripts/export_onnx.py \
  --ckpt ./artifacts/baseline.pt \
  --output ./artifacts/baseline.onnx \
  --sample_file data/raw/ESC-50-master/audio/1-19111-A-24.wav

python scripts/bench_onnx.py \
  --onnx ./artifacts/baseline.onnx \
  --file data/raw/ESC-50-master/audio/1-19111-A-24.wav

## Personal Note
I built this to practice shipping end-to-end ML: data wrangling, feature design, training, federated sim, and deployability (ONNX + latency). I made a few opinionated choices (2s windows, 64 mels, TinyCNN ~150k params) to keep it fast on CPU. If you have ideas to stress-test thresholds or try DP-SGD, open an issue—happy to iterate.
— Jaideep (@jaidoescode)

