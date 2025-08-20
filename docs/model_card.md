# Model Card — Cough vs Not-Cough (ESC-50 subset)
Intended use: demo of privacy-aware, edge-oriented audio classification.
Data: ESC-50; positives = "cough/coughing"; negatives = others; 2s @16kHz → 64-mel log-mels.
Training: TinyCNN (~150k params). Centralized baseline; FedAvg simulation.
Eval: Acc/F1/AUC on small test split (see artifacts/*_eval.json).
Limits: small test set; not clinical; not for medical use.
Privacy: public data only; federated sim = illustrative.
