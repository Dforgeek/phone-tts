from pathlib import Path
import sys
import argparse
import json
import time
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from scipy.io.wavfile import write

PROJECT_ROOT = Path("/Users/egorkolesnikv/Documents/ai_talent/edl/vosk").resolve()
VITS2_PATH = PROJECT_ROOT / "vosk-tts" / "training" / "vits2"
if str(VITS2_PATH) not in sys.path:
    sys.path.insert(0, str(VITS2_PATH))

import models
import utils
import text
import commons
from text.symbols import symbols


def build_model(config: dict) -> nn.Module:
    model = models.SynthesizerTrn(
        len(symbols),
        80,
        config["train"]["segment_size"] // config["data"]["hop_length"],
        n_speakers=config["data"]["n_speakers"],
        mas_noise_scale_initial=0.01,
        noise_scale_delta=2e-6,
        **config["model"],
    ).cpu()
    return model


def get_text_tensor(input_text: str, cfg: dict) -> torch.LongTensor:
    seq = text.text_to_sequence_g2p(input_text)
    if cfg["data"]["add_blank"]:
        seq = commons.intersperse(seq, 0)
    return torch.LongTensor(seq)


def run_full_torch(model: nn.Module, config: dict, text_str: str, speaker: int, torch_trace: Path, out_wav: Path):
    device = "cpu"
    tokens = get_text_tensor(text_str, config)
    x_tst = tokens.to(device).unsqueeze(0)
    lengths = torch.LongTensor([tokens.size(0)]).to(device)
    sid = torch.LongTensor([speaker]).to(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.no_grad():
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            t0 = time.perf_counter()
            audio = (
                model.infer(
                    x_tst,
                    lengths,
                    sid=sid,
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1.0,
                )[0][0, 0]
                .data.cpu()
                .numpy()
                * 32768.0
            )
            t1 = time.perf_counter()
    torch_trace.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(torch_trace))
    write(str(out_wav), config["data"]["sampling_rate"], audio.astype(np.int16))
    return t1 - t0


def run_ort_torch(model: nn.Module, config: dict, encoder_onnx: Path, text_str: str, speaker: int,
                  torch_trace: Path, ort_profile_dir: Path, out_wav: Path):
    device = "cpu"
    tokens = get_text_tensor(text_str, config)
    tokens_b = tokens.unsqueeze(0)  # [1, T]
    lengths = torch.LongTensor([tokens.size(0)])     # [1]
    sid = torch.LongTensor([speaker])                # [1]

    # ORT session with profiling
    so = ort.SessionOptions()
    so.enable_profiling = True
    sess = ort.InferenceSession(str(encoder_onnx), sess_options=so, providers=["CPUExecutionProvider"])
    ort_inputs = {
        sess.get_inputs()[0].name: tokens_b.numpy().astype(np.int64),
        sess.get_inputs()[1].name: lengths.numpy().astype(np.int64),
        sess.get_inputs()[2].name: sid.numpy().astype(np.int64),
    }
    t_ort0 = time.perf_counter()
    x_np, m_p_np, logs_p_np, x_mask_np = sess.run(None, ort_inputs)
    t_ort1 = time.perf_counter()
    ort_prof_file = sess.end_profiling()
    # Move ORT profile to desired dir
    ort_profile_dir.mkdir(parents=True, exist_ok=True)
    ort_prof_dst = ort_profile_dir / Path(ort_prof_file).name
    Path(ort_prof_file).replace(ort_prof_dst)

    # Convert encoder outputs to torch
    x = torch.from_numpy(x_np).to(dtype=torch.float32)
    m_p = torch.from_numpy(m_p_np).to(dtype=torch.float32)
    logs_p = torch.from_numpy(logs_p_np).to(dtype=torch.float32)
    x_mask = torch.from_numpy(x_mask_np).to(dtype=torch.float32)

    if model.n_speakers > 0:
        g = model.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
    else:
        g = None

    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.no_grad():
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            t_torch0 = time.perf_counter()
            if model.use_sdp:
                logw = model.dp(x, x_mask, g=g, reverse=True, noise_scale=0.8)
            else:
                logw = model.dp(x, x_mask, g=g)
            w = torch.exp(logw) * x_mask * 1.0
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = commons.generate_path(w_ceil, attn_mask)

            m_p_exp = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
            logs_p_exp = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

            z_p = m_p_exp + torch.randn_like(m_p_exp) * torch.exp(logs_p_exp) * 0.667
            z = model.flow(z_p, y_mask, g=g, reverse=True)
            o, o_mb = model.dec((z * y_mask)[:, :, :None], g=g)
            audio = o[0, 0].data.cpu().numpy() * 32768.0
            t_torch1 = time.perf_counter()

    prof.export_chrome_trace(str(torch_trace))
    write(str(out_wav), config["data"]["sampling_rate"], audio.astype(np.int16))
    return (t_ort1 - t_ort0), (t_torch1 - t_torch0), ort_prof_dst


def main():
    parser = argparse.ArgumentParser(description="Compare profiling: Torch vs ORT+Torch")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "pretrained" / "config.json"))
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "pretrained" / "G_1000.pth"))
    parser.add_argument("--encoder_onnx", type=str, default=str(PROJECT_ROOT / "pretrained" / "encoder.int8.onnx"))
    parser.add_argument("--text", type=str, default="С другой стороны постоянный количественный рост и сфера нашей активности представляет собой интересный эксперимент проверки направлений прогрессивного развития.")
    parser.add_argument("--speaker", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT))
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    out_dir = Path(args.out_dir)
    (out_dir / "profiles").mkdir(parents=True, exist_ok=True)

    # Build once
    model = build_model(config).cpu().eval()
    utils.load_checkpoint(args.checkpoint, model, None)
    model.eval()

    # Full Torch
    t_torch = run_full_torch(
        model, config, args.text, args.speaker,
        torch_trace=out_dir / "profiles" / "infer_profile_torch.json",
        out_wav=out_dir / "out_torch.wav",
    )

    # ORT Encoder + Torch Decoder
    t_ort, t_torch_rest, ort_prof = run_ort_torch(
        model, config, Path(args.encoder_onnx), args.text, args.speaker,
        torch_trace=out_dir / "profiles" / "infer_profile_ort_torch.json",
        ort_profile_dir=out_dir / "profiles",
        out_wav=out_dir / "out_ort_torch.wav",
    )

    total_ort_torch = t_ort + t_torch_rest
    print("=== Timing (seconds) ===")
    print(f"Torch full: {t_torch:.4f}")
    print(f"ORT encoder: {t_ort:.4f}, Torch rest: {t_torch_rest:.4f}, Total ORT+Torch: {total_ort_torch:.4f}")
    print("Profiles written to:", out_dir / "profiles")
    print("ORT session profile:", ort_prof)


if __name__ == "__main__":
    main()




