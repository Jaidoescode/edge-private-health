# tests/test_minimal.py
import torch
from src.models.tiny_cnn import TinyCNN
from src.federated.fed_utils import aggregate

def test_forward_shape():
    model = TinyCNN()
    x = torch.randn(2, 1, 64, 128)  # [B, C, n_mels, T]
    y = model(x)
    assert y.shape == (2, 2)

def test_fedavg_dtype_safe():
    model = TinyCNN()
    sd1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sd2 = {k: v.detach().clone() * 0.0 for k, v in model.state_dict().items()}
    agg = aggregate([sd1, sd2])  # must not crash on Long tensors
    # sanity: at least one tensor is floating-point
    assert any(v.dtype.is_floating_point for v in agg.values())
