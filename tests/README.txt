Edge-Private Health — Cough Detection (ESC-50)

CI Status: See GitHub Actions workflow in .github/workflows/ci.yml

Disclaimer:
This project is for research/demo purposes only.
Not intended for medical use.

--------------------------------------------------------------------------------
Overview
--------------------------------------------------------------------------------
This project is a privacy-aware machine learning demo that detects
cough vs. not-cough from short audio clips.

It highlights end-to-end ML skills with production-minded practices:

- Data pipeline — ESC-50 subset, log-mel feature extraction
- Modeling — TinyCNN baseline classifier
- Federated Learning (FedAvg) — simulates decentralized, privacy-preserving training
- Evaluation — accuracy, F1, ROC-AUC
- Deployment — ONNX export + CPU benchmarking
- Reproducibility — CI with tests

--------------------------------------------------------------------------------
Quickstart
--------------------------------------------------------------------------------
# 0) (optional) create a venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Download and prepare ESC-50 cough vs. non-cough subset
python -m src.data.prepare --root ./data --seed 42

# 2) Train centralized baseline
python -m src.train.train_baseline --data_root ./data --epochs 10

# 3) (Optional) Federated training simulation
python -m src.federated.fed_sim --data_root ./data --clients 5 --rounds 3 --local_epochs 1

# 4) Evaluate models
python -m src.eval.report --data_root ./data --ckpt ./artifacts/baseline.pt
python -m src.eval.report --data_root ./data --ckpt ./artifacts/fedavg.pt

# 5) Run inference on a sample audio
python -m src.infer --file data/raw/ESC-50-master/audio/1-19111-A-24.wav --ckpt ./artifacts/baseline.pt

--------------------------------------------------------------------------------
Results
--------------------------------------------------------------------------------
Model Results:
- Baseline (CNN): Accuracy=0.67, F1=0.60, AUC=1.00
- Federated Avg: Accuracy=0.25, F1=0.40, AUC=0.74

ONNX Benchmark (CPU):
- Mean latency: 0.71 ms
- P50 latency: 0.66 ms
- P90 latency: 0.79 ms
- P99 latency: 1.76 ms
- Input shape: [1, 1, 64, 201]

--------------------------------------------------------------------------------
Tech Stack
--------------------------------------------------------------------------------
- Python 3.9
- PyTorch 2.8.0
- torchaudio 2.8.0
- scikit-learn 1.6.1
- matplotlib, tqdm, pydantic
- ONNX / onnxruntime
- GitHub Actions (CI)

--------------------------------------------------------------------------------
Reproducibility & CI
--------------------------------------------------------------------------------
- Tests: run via "pytest -q"
- CI: automatic check on every push
- Determinism: fixed seeds for data prep & training splits

--------------------------------------------------------------------------------
Repo Structure
--------------------------------------------------------------------------------
edge-private-health/
├── src/                # source code
│   ├── data/           # data prep
│   ├── train/          # baseline training
│   ├── federated/      # FedAvg simulation
│   ├── eval/           # evaluation
│   └── models/         # TinyCNN
├── scripts/            # ONNX export + benchmark
├── artifacts/          # trained models + exports
├── tests/              # pytest unit tests
├── docs/               # project docs, model card
├── requirements.txt
├── README.md
└── .github/workflows/  # CI config

--------------------------------------------------------------------------------
Motivation
--------------------------------------------------------------------------------
- Real-world healthcare AI requires privacy and deployability.
- This project demonstrates:
  * Privacy-aware training with federated learning (no raw data sharing).
  * Edge readiness with ONNX export and CPU latency benchmarks.
- While the model is small and dataset limited, the pipeline shows how you’d
  structure a production ML project.

--------------------------------------------------------------------------------
License
--------------------------------------------------------------------------------
MIT License — see LICENSE

--------------------------------------------------------------------------------
About
--------------------------------------------------------------------------------
Built and maintained by Jaideep Singh (github.com/jaidoescode).
For questions, open an Issue on GitHub.
