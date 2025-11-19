#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch

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
from models import SynthesizerTrn  # noqa: E402
from text.symbols import symbols  # noqa: E402

# PTQ helpers (from exp/egor/ptq.py)
from exp.egor.ptq import (  # noqa: E402
    prepare_model_for_ptq_convs_only,
    convert_model_from_ptq,
)

from torch.nn.utils.weight_norm import WeightNorm, remove_weight_norm  # noqa: E402


def build_model(hps, is_onnx: bool = True) -> SynthesizerTrn:
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


def make_infer_forward(net_g: SynthesizerTrn):
    def infer_forward(text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio = net_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )[0].unsqueeze(1)
        return audio

    return infer_forward


def try_export_onnx(
    net_g: SynthesizerTrn,
    out_path: str,
    opset: int,
):
    # Dummy shapes similar to onnx_export.py
    num_symbols = net_g.n_vocab
    dmy_text = torch.randint(low=0, high=num_symbols, size=(1, 50), dtype=torch.long)
    dmy_text_length = torch.LongTensor([dmy_text.size(1)])
    dmy_scales = torch.FloatTensor([0.667, 1.0, 0.8])  # noise, length, noise_w
    dmy_sid = torch.LongTensor([0])

    torch.onnx.export(
        model=net_g,
        args=(dmy_text, dmy_text_length, dmy_scales, dmy_sid),
        f=out_path,
        verbose=False,
        opset_version=opset,
        dynamo=False,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        },
    )

def strip_weight_norm_hooks(model: torch.nn.Module) -> None:
    """
    Remove any WeightNorm parametrizations and dangling pre-hooks,
    which can break TorchScript/JIT tracing used by ONNX export.
    """
    for m in model.modules():
        try:
            remove_weight_norm(m)
        except (ValueError, AttributeError):
            pass
        if hasattr(m, "_forward_pre_hooks"):
            for k, hook in list(m._forward_pre_hooks.items()):
                if isinstance(hook, WeightNorm):
                    del m._forward_pre_hooks[k]


def main():
    parser = argparse.ArgumentParser(description="Export VITS2 model to ONNX (try quantized export).")
    parser.add_argument("--config", required=True, type=str, help="Path to config.json")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to checkpoint (float or quantized)")
    parser.add_argument("--quantized", action="store_true", help="Treat checkpoint as INT8 quantized; reconstruct quant graph before loading")
    parser.add_argument("--speaker-id", default=0, type=int, help="Default speaker id for dummy input")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "model.onnx"), type=str, help="Output ONNX file")
    parser.add_argument("--opset", default=15, type=int, help="ONNX opset version")
    parser.add_argument("--try-float-fallback", action="store_true", help="If quantized export fails, fall back to float export")
    args = parser.parse_args()

    os.makedirs(str(Path(args.out).parent), exist_ok=True)

    # Load hparams
    hps = utils.get_hparams_from_file(args.config)

    # Build model
    if args.quantized:
        # Rebuild quantized structure (Conv/ConvT/Linear) and load state
        net_g = build_model(hps, is_onnx=True)
        try:
            net_g.dec.remove_weight_norm()
        except Exception:
            pass
        try:
            net_g.flow.remove_weight_norm()
        except Exception:
            pass
        prepare_model_for_ptq_convs_only(net_g, module_roots=None, backend="auto")
        convert_model_from_ptq(net_g)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        missing, unexpected = net_g.load_state_dict(checkpoint["model"], strict=False)
        if missing or unexpected:
            print(f"MISSING: {missing}")
            print(f"UNEXPECTED: {unexpected}")
        net_g.eval()
    else:
        net_g = build_model(hps, is_onnx=True)
        utils.load_checkpoint(args.checkpoint, net_g, None)
        net_g.eval()

    # Prepare forward wrapper for export
    net_g.forward = make_infer_forward(net_g)
    # Ensure no residual WeightNorm hooks interfere with tracing/export
    strip_weight_norm_hooks(net_g)

    # Try export
    try:
        try_export_onnx(net_g, args.out, args.opset)
        print(f"Exported ONNX to: {args.out}")
        return
    except Exception as e:
        print(f"Quantized export failed: {e}")
        if not args.try_float_fallback:
            raise

    # Fallback to float export if requested
    print("Falling back to float export...")
    net_g_f = build_model(hps, is_onnx=True)
    utils.load_checkpoint(args.checkpoint, net_g_f, None) if not args.quantized else utils.load_checkpoint(args.checkpoint, net_g_f, None)
    try:
        net_g_f.dec.remove_weight_norm()
    except Exception:
        pass
    try:
        net_g_f.flow.remove_weight_norm()
    except Exception:
        pass
    net_g_f.eval()
    net_g_f.forward = make_infer_forward(net_g_f)
    strip_weight_norm_hooks(net_g_f)
    try_export_onnx(net_g_f, args.out, args.opset)
    print(f"Exported ONNX (float fallback) to: {args.out}")


if __name__ == "__main__":
    main()


