# scripts/export_onnx.py
import argparse, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from pathlib import Path
import soundfile as sf
import numpy as np

from src.models.tiny_cnn import TinyCNN
from src.data.prepare import wav_to_logmel

def load_wav(path: str):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    else:
        audio = audio.T
    return torch.from_numpy(audio), sr

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--output", required=True, help="Path to save .onnx file")
    ap.add_argument("--sample_file", default=None, help="Optional .wav for shape")
    args = ap.parse_args()

    model = TinyCNN()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    if args.sample_file and Path(args.sample_file).exists():
        wav, sr = load_wav(args.sample_file)
        dummy = wav_to_logmel(wav, sr).unsqueeze(0)  # [1,1,64,T]
    else:
        dummy = torch.randn(1, 1, 64, 44)  # fallback

    torch.onnx.export(
        model, dummy, args.output,
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["logmel"], output_names=["logits"],
        dynamic_axes={"logmel": {0: "batch_size", 3: "time"}}
    )
    print(f"✅ Exported model to {args.output}")
