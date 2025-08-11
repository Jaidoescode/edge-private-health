
import copy, torch, math, random
from torch.utils.data import DataLoader, TensorDataset

def split_dataset_iid(X, y, clients:int):
    N = X.shape[0]
    idx = list(range(N)); random.shuffle(idx)
    parts = [idx[i::clients] for i in range(clients)]
    return [(X[p], y[p]) for p in parts]

def local_train(model, data, epochs:int=1, lr:float=1e-3, dp:bool=False, clip:float=1.0, noise:float=0.1, device="cpu", batch_size=32):
    X, y = data
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    model = copy.deepcopy(model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            if dp:
                # Toy per-sample gradient: use grad of mean loss approximated with clipping + noise at parameter level
                loss.backward()
                # Clip grads
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
                # Add Gaussian noise
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad += noise * torch.randn_like(p.grad)
                opt.step()
            else:
                loss.backward(); opt.step()
    return model.state_dict()

def aggregate(state_dicts):
    # FedAvg: average only floating tensors; copy ints/bools from the first client
    import copy, torch
    agg = copy.deepcopy(state_dicts[0])
    for k in agg.keys():
        if torch.is_floating_point(agg[k]):
            # start with zeros of same shape/dtype/device
            summed = torch.zeros_like(agg[k])
            for sd in state_dicts:
                summed = summed + sd[k].to(summed.dtype)
            agg[k] = summed / float(len(state_dicts))
        else:
            # e.g., num_batches_tracked (Long), or buffers that shouldn't be averaged
            agg[k] = state_dicts[0][k]
    return agg

