# scripts/export_onnx.py
import argparse
import torch
from src.models.tiny_cnn import TinyCNN
from src.data.prepare import wav_to_logmel
import soundfile as sf
import numpy as np
from pathlib import Path

def load_wav(path: str):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    else:
        audio = audio.T
    wav = torch.from_numpy(audio)
    return wav, sr

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--output", required=True, help="Path to save .onnx file")
    ap.add_argument("--sample_file", default=None, help="Optional sample .wav for shape inference")
    args = ap.parse_args()

    # Load model
    model = TinyCNN()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    # Build a dummy input with the correct shape
    if args.sample_file and Path(args.sample_file).exists():
        wav, sr = load_wav(args.sample_file)
        dummy = wav_to_logmel(wav, sr).unsqueeze(0)  # [1,1,64,T]
    else:
        # default fallback: [1,1,64,44] (44 is arbitrary; model is time-agnostic)
        dummy = torch.randn(1, 1, 64, 44)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy,
        args.output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["logmel"],
        output_names=["logits"],
        dynamic_axes={"logmel": {0: "batch_size", 3: "time"}}
    )
    print(f"✅ Exported model to {args.output}")
