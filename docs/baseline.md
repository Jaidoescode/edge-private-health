# Baseline Results (MVP)

**Task:** ESC-50 subset — “cough” vs “not-cough”  
**Input:** 2s log-mel (64 mels, 16 kHz)  
**Model:** TinyCNN (~150k params)

**Centralized baseline (best checkpoint)**
- Acc: 0.667
- F1:  0.600
- AUC: 1.000

**Federated (FedAvg, 5 clients × 3 rounds, 1 local epoch)**
- Acc: 0.250
- F1:  0.400
- AUC: 0.741

Notes:
- Very small test set ⇒ accuracy is coarse; AUC is the better signal.
- This is a one-day MVP; next iterations: Opacus DP-SGD (ε accounting), quantization + Pi deploy, secure aggregation sim.
