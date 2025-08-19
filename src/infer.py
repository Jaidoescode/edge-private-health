import argparse
import pathlib
import torch
import numpy as np
import soundfile as sf

from src.data.prepare import wav_to_logmel, SAMPLE_RATE
from src.models.tiny_cnn import TinyCNN


def load_wav(path: str) -> tuple[torch.Tensor, int]:
    """Load a WAV file as a Tensor shaped [C, N] with float32 samples."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]  # [1, N]
    else:
        audio = audio.T               # [C, N]
    wav = torch.from_numpy(audio)
    return wav, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to a .wav file")
    ap.add_argument("--ckpt", default="./artifacts/baseline.pt",
                    help="Path to a trained checkpoint (.pt)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold for classifying as COUGH (default: 0.5)")
    args = ap.parse_args()

    # Load model
    model = TinyCNN()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    # Load audio and preprocess exactly like training
    wav, sr = load_wav(args.file)
    logmel = wav_to_logmel(wav, sr).unsqueeze(0)  # [1,1,64,T]

    with torch.no_grad():
        logits = model(logmel)
        probs = torch.softmax(logits, dim=1)[0]
        p_not, p_cough = probs.tolist()

    pred_label = "COUGH" if p_cough >= args.threshold else "NOT-COUGH"

    print(f"File: {args.file}")
    print(f"Prob not-cough: {p_not:.3f}")
    print(f"Prob cough:     {p_cough:.3f}")
    print(f"Threshold:      {args.threshold:.2f}")
    print("Pred:", pred_label)


if __name__ == "__main__":
    main()
