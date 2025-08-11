
import argparse, os, torch, json
from torch.utils.data import TensorDataset, DataLoader
from src.models.tiny_cnn import TinyCNN
from src.federated.fed_utils import split_dataset_iid, local_train, aggregate
from src.train.train_baseline import load_split, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--dp", action="store_true", help="enable toy DP-SGD (clip+noise)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, ytr = load_split(args.data_root, "train")
    Xva, yva = load_split(args.data_root, "val")
    Xte, yte = load_split(args.data_root, "test")

    clients_data = split_dataset_iid(Xtr, ytr, args.clients)

    global_model = TinyCNN().to(device)
    for r in range(1, args.rounds+1):
        client_states = []
        for cid, data in enumerate(clients_data):
            sd = local_train(global_model, data, epochs=args.local_epochs, lr=1e-3, dp=args.dp, clip=args.clip, noise=args.noise, device=device)
            client_states.append(sd)
        new_state = aggregate(client_states)
        global_model.load_state_dict(new_state)
        val_metrics = evaluate(global_model, DataLoader(TensorDataset(Xva, yva), batch_size=128), device)
        print(f"Round {r}: val={val_metrics}")

    # Final test
    test_metrics = evaluate(global_model, DataLoader(TensorDataset(Xte, yte), batch_size=128), device)
    print("Test:", test_metrics)
    # save
    import pathlib, json
    art = pathlib.Path("./artifacts"); art.mkdir(exist_ok=True)
    torch.save(global_model.state_dict(), art / "fedavg.pt")
    (art / "fedavg_metrics.json").write_text(json.dumps(test_metrics, indent=2))

if __name__ == "__main__":
    main()
