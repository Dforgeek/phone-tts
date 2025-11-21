#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from scipy.io.wavfile import write

# Resolve project root and add required paths for imports
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
VITS2_DIR = PROJECT_ROOT / "vosk-tts" / "training" / "vits2"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(VITS2_DIR) not in sys.path:
    sys.path.append(str(VITS2_DIR))

# VITS2 imports
import utils  # noqa: E402
import commons  # noqa: E402
import text as text_mod  # noqa: E402
from models import SynthesizerTrn  # noqa: E402
from text.symbols import symbols  # noqa: E402

# PTQ helpers (from exp/egor/ptq.py)
from exp.egor.ptq import (  # noqa: E402
    prepare_model_for_ptq_convs_only,
    convert_model_from_ptq,
)


def build_model(hps, is_onnx: bool = False) -> SynthesizerTrn:
    if getattr(hps.model, "use_mel_posterior_encoder", False):
        posterior_channels = 80
        hps.data.use_mel_posterior_encoder = True
    else:
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        is_onnx=is_onnx,
        **vars(hps.model),
    ).to("cpu")
    return net_g


def get_text_tensor(txt: str, hps) -> torch.LongTensor:
    # Follow the notebook path: use g2p + add_blank
    seq = text_mod.text_to_sequence_g2p(txt)
    if getattr(hps.data, "add_blank", True):
        seq = commons.intersperse(seq, 0)
    return torch.LongTensor(seq)


def vcss(
    model: SynthesizerTrn,
    txt: str,
    hps,
    speaker_id: int,
    out_wav_path: str = "",
):
    stn_tst = get_text_tensor(txt, hps)
    speed = 1.0
    sid = torch.LongTensor([speaker_id]).to("cpu")
    with torch.no_grad():
        x_tst = stn_tst.to("cpu").unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to("cpu")
        audio = (
            model.infer(
                x_tst,
                x_tst_lengths,
                sid=sid,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1 / speed,
            )[0][0, 0]
            .cpu()
            .numpy()
            * 32768.0
        )
    if out_wav_path:
        write(out_wav_path, hps.data.sampling_rate, audio.astype(np.int16))
    return audio


def main():
    parser = argparse.ArgumentParser(description="Profile VITS2 inference (FP32 or INT8).")
    parser.add_argument("--config", required=True, type=str, help="Path to config.json")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to checkpoint (float or INT8 state dict)")
    parser.add_argument("--quantized", action="store_true", help="Treat checkpoint as INT8 quantized and reconstruct quant graph")
    parser.add_argument("--text", default="С трев+ожным ч+увством бер+усь я з+а пер+о.", type=str, help="Text to synthesize")
    parser.add_argument("--speaker-id", default=1, type=int, help="Speaker id to use")
    parser.add_argument("--repeat", default=5, type=int, help="Number of runs to profile")
    parser.add_argument("--save-wav", default="", type=str, help="If set, save last run to this wav file")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "output"), type=str, help="Output directory for wav (if save-wav not absolute)")
    parser.add_argument("--trace", default="", type=str, help="If set, write chrome trace json to this path")
    args = parser.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(args.out_dir, exist_ok=True)

    # Load hparams
    hps = utils.get_hparams_from_file(args.config)
    # Match inference text processing to notebook
    hps.data.aligned_text = False
    hps.data.g2p_text = True

    # Build model
    if args.quantized:
        # Build float skeleton, prepare quant wrappers, convert, then load INT8 weights
        net_g = build_model(hps)
        try:
            net_g.dec.remove_weight_norm()
        except Exception:
            pass
        try:
            net_g.flow.remove_weight_norm()
        except Exception:
            pass
        # Prepare the same way as during quantization (whole model conv/linear)
        prepare_model_for_ptq_convs_only(net_g, module_roots=None, backend="auto")
        convert_model_from_ptq(net_g)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        net_g.load_state_dict(checkpoint["model"], strict=False)
        net_g.eval()
    else:
        net_g = build_model(hps)
        utils.load_checkpoint(args.checkpoint, net_g, None)
        net_g.eval().to("cpu")

    # Profile
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("vcss_inference"):
            for _ in range(max(1, args.repeat)):
                out_path = ""
                if args.save_wav:
                    if os.path.isabs(args.save_wav):
                        out_path = args.save_wav
                    else:
                        out_path = os.path.join(args.out_dir, args.save_wav)
                _ = vcss(net_g, args.text, hps, args.speaker_id, out_wav_path=out_path)

    if args.trace:
        prof.export_chrome_trace(args.trace)

    # Show top operators by self CPU time
    print(
        prof.key_averages(group_by_stack_n=10).table(
            sort_by="self_cpu_time_total", row_limit=25
        )
    )


if __name__ == "__main__":
    main()



