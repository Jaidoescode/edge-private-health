
import argparse, os, pathlib, torch, json
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from src.models.tiny_cnn import TinyCNN

def load_split(data_root, split):
    d = torch.load(os.path.join(data_root, "processed", f"{split}.pt"))
    return d["X"].float(), d["y"].long()

def train_epoch(model, loader, optim, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total = 0; correct = 0; loss_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        logits = model(xb)
        loss = ce(logits, yb)
        loss.backward()
        optim.step()
        loss_sum += loss.item() * xb.size(0)
        total += xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob1 = torch.softmax(logits, dim=1)[:,1].cpu()
        ys.append(yb)
        ps.append(prob1)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    acc = accuracy_score(y, (p>=0.5).astype(int))
    f1 = f1_score(y, (p>=0.5).astype(int))
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "auc": auc}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, ytr = load_split(args.data_root, "train")
    Xva, yva = load_split(args.data_root, "val")
    Xte, yte = load_split(args.data_root, "test")

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=128)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=128)

    model = TinyCNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0; best_path = pathlib.Path("./artifacts"); best_path.mkdir(exist_ok=True)
    for ep in range(1, args.epochs+1):
        loss, acc = train_epoch(model, train_loader, optim, device)
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {ep}: loss={loss:.4f} train_acc={acc:.3f} val={val_metrics}")
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path / "baseline.pt")

    # final test
    model.load_state_dict(torch.load(best_path / "baseline.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    (best_path / "baseline_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print("Test:", test_metrics)

if __name__ == "__main__":
    main()
