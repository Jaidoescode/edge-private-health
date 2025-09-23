# scripts/bench_onnx.py
import argparse, sys, pathlib, time, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from src.data.prepare import wav_to_logmel

def load_input(wav_path: str):
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    else:
        audio = audio.T
    wav = torch.from_numpy(audio)
    x = wav_to_logmel(wav, sr).unsqueeze(0).numpy().astype(np.float32)  # [1,1,64,T]
    return x

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--file", required=True, help="WAV used to build input tensor")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()

    x = load_input(args.file)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    # warmup
    for _ in range(args.warmup):
        _ = sess.run([out], {inp: x})

    ts = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        _ = sess.run([out], {inp: x})
        ts.append((time.perf_counter() - t0) * 1000.0)

    stats = {
        "onnx": args.onnx,
        "runs": args.runs,
        "ms_mean": float(np.mean(ts)),
        "ms_p50": float(np.percentile(ts, 50)),
        "ms_p90": float(np.percentile(ts, 90)),
        "ms_p99": float(np.percentile(ts, 99)),
        "input_shape": list(x.shape),
    }
    print(json.dumps(stats, indent=2))
