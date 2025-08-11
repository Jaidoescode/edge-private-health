
import argparse, os, json, torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.tiny_cnn import TinyCNN
from src.train.train_baseline import load_split, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xte, yte = load_split(args.data_root, "test")
    loader = DataLoader(TensorDataset(Xte, yte), batch_size=128)

    model = TinyCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    metrics = evaluate(model, loader, device)
    print("Eval:", metrics)
    out = os.path.splitext(args.ckpt)[0] + "_eval.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Wrote", out)

if __name__ == "__main__":
    main()
