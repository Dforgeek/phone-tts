#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write
import onnxruntime as ort
import time

# Resolve project root and add VITS2 training code to path (for text preprocessing and config)
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
VITS2_DIR = PROJECT_ROOT / "vosk-tts" / "training" / "vits2"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(VITS2_DIR) not in sys.path:
    sys.path.append(str(VITS2_DIR))

import utils  # noqa: E402
import commons  # noqa: E402
import text as text_mod  # noqa: E402


def text_to_ids(txt: str, add_blank: bool) -> np.ndarray:
    ids = text_mod.text_to_sequence_g2p(txt)
    if add_blank:
        ids = commons.intersperse(ids, 0)
    return np.asarray(ids, dtype=np.int64)


def run_onnx(
    onnx_path: str,
    config_path: str,
    txt: str,
    speaker_id: int,
    out_wav: str,
    noise: float = 0.667,
    length: float = 1.0,
    noise_w: float = 0.8,
) -> None:
    # Load config for sampling rate and add_blank
    hps = utils.get_hparams_from_file(config_path)
    add_blank = getattr(hps.data, "add_blank", True)
    sampling_rate = hps.data.sampling_rate

    # Prepare inputs
    ids = text_to_ids(txt, add_blank=add_blank)
    input_np = ids[None, :]  # [1, T]
    input_lengths_np = np.array([ids.shape[0]], dtype=np.int64)  # [1]
    scales_np = np.array([noise, length, noise_w], dtype=np.float32)  # [3]
    sid_np = np.array([speaker_id], dtype=np.int64)  # [1]

    # ONNX Runtime session
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)

    # Run
    t0 = time.perf_counter()
    outputs = sess.run(
        None,
        {
            "input": input_np,
            "input_lengths": input_lengths_np,
            "scales": scales_np,
            "sid": sid_np,
        },
    )
    dt = time.perf_counter() - t0
    y = outputs[0]  # expect shape [B, 1, T]
    if y.ndim == 3:
        y = y[0, 0]
    elif y.ndim == 2:
        y = y[0]
    # Report timing/RTF
    audio_sec = float(y.shape[-1]) / float(sampling_rate)
    rtf = dt / audio_sec if audio_sec > 0 else float("inf")
    print(f"Inference time: {dt*1000:.1f} ms  |  audio: {audio_sec:.2f} s  |  RTF: {rtf:.3f}")
    y = (y * 32768.0).astype(np.int16)

    # Save
    out_path = Path(out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), sampling_rate, y)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run VITS2 ONNX inference (CPU).")
    parser.add_argument("--onnx", required=True, type=str, help="Path to ONNX model")
    parser.add_argument("--config", required=True, type=str, help="Path to config.json used to train/export")
    parser.add_argument("--text", required=False, type=str, default="С трев+ожным ч+увством бер+усь я з+а пер+о.", help="Input text")
    parser.add_argument("--speaker-id", type=int, default=0, help="Speaker ID")
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "exp" / "egor" / "onnx_out.wav"), help="Output wav path")
    parser.add_argument("--noise", type=float, default=0.667, help="Noise scale")
    parser.add_argument("--length", type=float, default=1.0, help="Length scale")
    parser.add_argument("--noise-w", type=float, default=0.8, help="Noise w scale")
    args = parser.parse_args()

    run_onnx(
        onnx_path=args.onnx,
        config_path=args.config,
        txt=args.text,
        speaker_id=args.speaker_id,
        out_wav=args.out,
        noise=args.noise,
        length=args.length,
        noise_w=args.noise_w,
    )


if __name__ == "__main__":
    main()



