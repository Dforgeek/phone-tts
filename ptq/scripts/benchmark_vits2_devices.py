#!/usr/bin/env python3
"""
Benchmark VITS2 inference on CPU vs CUDA for configurable batch sizes.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

# Resolve project root and add required paths for imports
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
VITS2_DIR = PROJECT_ROOT / "vosk-tts" / "training" / "vits2"
for extra_path in (PROJECT_ROOT, VITS2_DIR):
    if str(extra_path) not in sys.path:
        sys.path.append(str(extra_path))

# VITS2 imports
import utils  # type: ignore  # noqa: E402
import commons  # type: ignore  # noqa: E402
import text as text_mod  # type: ignore  # noqa: E402
from models import SynthesizerTrn  # type: ignore  # noqa: E402
from text.symbols import symbols  # type: ignore  # noqa: E402

from exp.egor.ptq import (  # type: ignore  # noqa: E402
    prepare_model_for_ptq_convs_only,
    convert_model_from_ptq,
)


@dataclass
class Sample:
    text: str
    speaker_id: int


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
    ).eval()
    return net_g


def get_text_tensor(txt: str, hps) -> torch.LongTensor:
    seq = text_mod.text_to_sequence_g2p(txt)
    if getattr(hps.data, "add_blank", True):
        seq = commons.intersperse(seq, 0)
    return torch.LongTensor(seq)


def prepare_batch(
    samples: Sequence[Sample],
    hps,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    text_tensors = [get_text_tensor(sample.text, hps) for sample in samples]
    lengths = torch.LongTensor([tensor.size(0) for tensor in text_tensors])
    max_len = int(lengths.max().item())
    padded = torch.zeros(len(text_tensors), max_len, dtype=torch.long)
    for idx, tensor in enumerate(text_tensors):
        padded[idx, : tensor.size(0)] = tensor
    speaker_ids = torch.LongTensor([sample.speaker_id for sample in samples])
    return padded, lengths, speaker_ids


def load_samples(
    list_path: Path,
    text_column: int,
    speaker_column: int,
    fallback_text: str,
) -> List[Sample]:
    samples: List[Sample] = []
    if list_path.exists():
        with open(list_path, "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip().split("|")
                if len(parts) <= max(text_column, speaker_column):
                    continue
                text = parts[text_column].strip()
                if not text:
                    continue
                try:
                    speaker = int(parts[speaker_column])
                except ValueError:
                    speaker = 0
                samples.append(Sample(text=text, speaker_id=speaker))
    if not samples:
        samples = [Sample(text=fallback_text, speaker_id=0)]
    return samples


def prepare_model(
    hps,
    checkpoint_path: Path,
    quantized: bool,
    backend: str = "auto",
) -> SynthesizerTrn:
    net_g = build_model(hps)
    if quantized:
        for attr in ("dec", "flow"):
            try:
                getattr(net_g, attr).remove_weight_norm()
            except Exception:
                pass
        prepare_model_for_ptq_convs_only(net_g, module_roots=None, backend=backend)
        convert_model_from_ptq(net_g)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        net_g.load_state_dict(checkpoint["model"], strict=False)
    else:
        utils.load_checkpoint(str(checkpoint_path), net_g, None)
    return net_g.eval()


def resolve_devices(requested: Sequence[str], quantized: bool) -> List[torch.device]:
    devices: List[torch.device] = []
    for spec in requested:
        name = spec.strip()
        if not name:
            continue
        try:
            device = torch.device(name)
        except RuntimeError:
            print(f"[warn] Skipping invalid device spec '{name}'")
            continue
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"[warn] CUDA not available, skipping '{name}'")
            continue
        if quantized and device.type != "cpu":
            print(f"[warn] Quantized torch.int8 model only supports CPU, skipping '{name}'")
            continue
        devices.append(device)
    # Deduplicate while preserving order
    deduped: List[torch.device] = []
    seen = set()
    for dev in devices:
        key = (dev.type, dev.index)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dev)
    if not deduped:
        raise RuntimeError("No valid devices resolved for benchmarking.")
    return deduped


def benchmark_device(
    device: torch.device,
    base_model: SynthesizerTrn,
    samples: List[Sample],
    hps,
    batch_size: int,
    repeats: int,
    warmup_runs: int,
    noise: float,
    length_scale: float,
    noise_w: float,
) -> None:
    model = base_model.to(device)
    sampling_rate = hps.data.sampling_rate
    total_runs = warmup_runs + repeats
    cursor = 0
    durations: List[float] = []
    rtf_values: List[float] = []

    def next_batch(start: int) -> List[Sample]:
        batch: List[Sample] = []
        for i in range(batch_size):
            idx = (start + i) % len(samples)
            batch.append(samples[idx])
        return batch

    for run_idx in range(total_runs):
        batch_samples = next_batch(cursor)
        cursor = (cursor + batch_size) % len(samples)
        text_batch, text_lengths, speaker_ids = prepare_batch(batch_samples, hps)
        text_batch = text_batch.to(device)
        text_lengths = text_lengths.to(device)
        speaker_ids = speaker_ids.to(device)

        sid = speaker_ids.view(-1)
        start_time = time.perf_counter()
        audio, _, _, y_mask, _ = model.infer(
            text_batch,
            text_lengths,
            sid=sid,
            noise_scale=noise,
            length_scale=length_scale,
            noise_scale_w=noise_w,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time

        with torch.no_grad():
            lengths = y_mask[:, 0, :].sum(dim=1).cpu().numpy()
        audio_total_sec = float(np.sum(lengths) / sampling_rate)

        if run_idx >= warmup_runs:
            durations.append(elapsed)
            if audio_total_sec > 0:
                rtf_values.append(elapsed / audio_total_sec)

        print(
            f"[{device}] run {run_idx + 1}/{total_runs} "
            f"time={elapsed * 1000:.1f} ms "
            f"audio_batch={audio_total_sec:.2f} s "
            f"{'(warmup)' if run_idx < warmup_runs else ''}"
        )

    if durations:
        print(f"\n== {device} stats ==")
        print(
            f"Latency per batch: mean={np.mean(durations)*1000:.1f} ms | "
            f"median={np.median(durations)*1000:.1f} ms | "
            f"min={np.min(durations)*1000:.1f} ms | "
            f"max={np.max(durations)*1000:.1f} ms"
        )
        if rtf_values:
            print(
                f"RTF: mean={np.mean(rtf_values):.3f} | "
                f"median={np.median(rtf_values):.3f} | "
                f"min={np.min(rtf_values):.3f} | "
                f"max={np.max(rtf_values):.3f}"
            )
    print("-" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark VITS2 inference on CPU vs CUDA with configurable batch size."
    )
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--quantized", action="store_true", help="Treat checkpoint as quantized PTQ state dict")
    parser.add_argument(
        "--dataset-list",
        default=str(PROJECT_ROOT / "natasha_dataset" / "audiopaths_sid_text.txt"),
        help="Path to audiopaths_sid_text.txt style file",
    )
    parser.add_argument("--text-column", type=int, default=3, help="Zero-based column for text in dataset list")
    parser.add_argument("--speaker-column", type=int, default=1, help="Zero-based column for speaker id")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for synthetic inference runs")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timed runs per device")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per device (not included in stats)")
    parser.add_argument("--devices", default="cpu,cuda", help="Comma-separated torch device specs (cpu,cuda,cuda:0,...)")
    parser.add_argument("--noise", type=float, default=0.667)
    parser.add_argument("--noise-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--fallback-text", type=str, default="С тревожным чувством берусь я за перо.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    random.seed(args.seed)

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)

    hps = utils.get_hparams_from_file(str(config_path))
    hps.data.aligned_text = False
    hps.data.g2p_text = True

    samples = load_samples(
        Path(args.dataset_list),
        text_column=args.text_column,
        speaker_column=args.speaker_column,
        fallback_text=args.fallback_text,
    )
    random.shuffle(samples)

    base_model = prepare_model(hps, checkpoint_path, args.quantized)

    requested_devices = [token.strip() for token in args.devices.split(",")]
    resolved_devices = resolve_devices(requested_devices, quantized=args.quantized)

    for device in resolved_devices:
        benchmark_device(
            device=device,
            base_model=prepare_model(hps, checkpoint_path, args.quantized) if device.type != "cpu" else base_model,
            samples=samples,
            hps=hps,
            batch_size=args.batch_size,
            repeats=args.repeats,
            warmup_runs=args.warmup,
            noise=args.noise,
            length_scale=args.length_scale,
            noise_w=args.noise_w,
        )


if __name__ == "__main__":
    main()


