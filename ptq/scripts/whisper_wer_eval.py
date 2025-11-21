#!/usr/bin/env python3
"""Evaluate TTS output quality using Whisper + WER."""
import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torchaudio.functional as AF
from scipy.io.wavfile import write

try:
    import whisper
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Please install openai-whisper: pip install -U openai-whisper") from exc

try:
    from jiwer import wer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Please install jiwer: pip install -U jiwer") from exc

# Resolve project root and make VITS2 training code importable
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
VITS2_DIR = PROJECT_ROOT / "vosk-tts" / "training" / "vits2"
for extra_path in (PROJECT_ROOT, VITS2_DIR):
    if str(extra_path) not in sys.path:
        sys.path.append(str(extra_path))

# VITS2 imports (after sys.path updates)
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
    idx: int
    reference_text: str
    speaker_id: int


def _build_model(hps, quantized: bool, checkpoint: str) -> SynthesizerTrn:
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
        is_onnx=False,
        **vars(hps.model),
    ).to("cpu")

    if quantized:
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
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        net_g.load_state_dict(checkpoint_data["model"], strict=False)
    else:
        utils.load_checkpoint(checkpoint, net_g, None)

    net_g.eval().to("cpu")
    return net_g


def _text_to_tensor(txt: str, hps) -> torch.LongTensor:
    seq = text_mod.text_to_sequence_g2p(txt)
    if getattr(hps.data, "add_blank", True):
        seq = commons.intersperse(seq, 0)
    return torch.LongTensor(seq)


def synthesize(
    model: SynthesizerTrn,
    text: str,
    hps,
    speaker_id: int,
    noise: float,
    length_scale: float,
    noise_w: float,
) -> np.ndarray:
    stn_tst = _text_to_tensor(text, hps)
    sid = torch.LongTensor([speaker_id]).to("cpu")
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to("cpu")
        x_tst_lens = torch.LongTensor([stn_tst.size(0)]).to("cpu")
        audio = (
            model.infer(
                x_tst,
                x_tst_lens,
                sid=sid,
                noise_scale=noise,
                noise_scale_w=noise_w,
                length_scale=max(1e-3, length_scale),
            )[0][0, 0]
            .cpu()
            .numpy()
        )
    return audio.astype(np.float32)


def load_dataset(list_path: str, text_column: int, speaker_column: int) -> List[Sample]:
    samples: List[Sample] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
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
            samples.append(Sample(idx=idx, reference_text=text, speaker_id=speaker))
    return samples


def pick_samples(
    samples: List[Sample],
    num_samples: int,
    strategy: str,
    seed: int,
) -> List[Sample]:
    if not samples:
        return []

    if strategy == "first":
        return samples[:num_samples]

    rng = random.Random(seed)
    cloned = samples.copy()
    rng.shuffle(cloned)
    return cloned[:num_samples]


def save_audio(audio: np.ndarray, path: Path, sampling_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    write(str(path), sampling_rate, wav)


def resample_for_whisper(audio: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    tensor = torch.from_numpy(audio).float()
    resampled = AF.resample(tensor, src_sr, dst_sr)
    return resampled.numpy()


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_GRAPHEME_MARK_RE = re.compile(r"\+")


def normalize_text(text: str) -> str:
    """
    Minimal normalization mirroring jiwer Compose pipeline (lower, strip punctuation, collapse whitespace).
    """
    lowered = text.lower()
    no_marks = _GRAPHEME_MARK_RE.sub("", lowered)
    no_punct = _PUNCT_RE.sub(" ", no_marks)
    normalized = " ".join(no_punct.split())
    return normalized.strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate TTS samples with Whisper + WER")
    parser.add_argument("--config", required=True, help="Path to VITS2 config.json")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (float or quantized)")
    parser.add_argument(
        "--dataset-list",
        default=str(PROJECT_ROOT / "natasha_dataset" / "audiopaths_sid_text.txt"),
        help="Path to audiopaths_sid_text-style file",
    )
    parser.add_argument("--text-column", type=int, default=3, help="Zero-based column index with reference text")
    parser.add_argument("--speaker-column", type=int, default=1, help="Zero-based column index with speaker id")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--sample-strategy", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--speaker-id", type=int, default=None, help="Override speaker id for all samples")
    parser.add_argument("--noise", type=float, default=0.667)
    parser.add_argument("--noise-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0, help="Length scale passed to VITS ( >1 slows speech )")
    parser.add_argument("--whisper-model", default="small", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--whisper-device", default=None, help="Device for Whisper (cpu/cuda)")
    parser.add_argument("--whisper-language", default="ru")
    parser.add_argument("--keep-wavs", action="store_true", help="Save synthesized wav files")
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "output" / "whisper_eval"),
        help="Directory for optional wav dumps and JSON report",
    )
    parser.add_argument("--report-json", default="", help="Optional path to save JSON report")
    parser.add_argument("--quantized", action="store_true", help="Treat checkpoint as quantized (PTQ int8)")
    args = parser.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))

    dataset_path = Path(args.dataset_list)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset list not found: {dataset_path}")

    samples = load_dataset(str(dataset_path), args.text_column, args.speaker_column)
    selected = pick_samples(samples, args.samples, args.sample_strategy, args.seed)
    if not selected:
        raise RuntimeError("No samples available for evaluation")

    hps = utils.get_hparams_from_file(args.config)
    hps.data.aligned_text = False
    hps.data.g2p_text = True

    model = _build_model(hps, args.quantized, args.checkpoint)
    sampling_rate = hps.data.sampling_rate

    whisper_device = args.whisper_device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = whisper_device != "cpu"
    whisper_model = whisper.load_model(args.whisper_model, device=whisper_device)

    reports = []

    out_dir = Path(args.out_dir)
    if args.keep_wavs:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(selected, 1):
        speaker = args.speaker_id if args.speaker_id is not None else sample.speaker_id
        audio = synthesize(
            model,
            sample.reference_text,
            hps,
            speaker,
            noise=args.noise,
            length_scale=args.length_scale,
            noise_w=args.noise_w,
        )
        if args.keep_wavs:
            save_audio(audio, out_dir / f"sample_{sample.idx:05d}.wav", sampling_rate)

        whisper_audio = resample_for_whisper(audio, sampling_rate, 16000)
        result = whisper_model.transcribe(
            whisper_audio,
            language=args.whisper_language,
            fp16=use_fp16,
            verbose=False,
            condition_on_previous_text=False,
        )
        hypothesis = result.get("text", "").strip()
        ref = sample.reference_text.strip()
        norm_ref = normalize_text(ref)
        norm_hyp = normalize_text(hypothesis)
        sample_wer = wer(norm_ref, norm_hyp)
        reports.append(
            {
                "sample_index": sample.idx,
                "reference": ref,
                "transcription": hypothesis,
                "speaker_id": speaker,
                "wer": sample_wer,
            }
        )
        print(f"[{i:02d}/{len(selected)}] idx={sample.idx} speaker={speaker} WER={sample_wer:.3f}")
        print(f"  REF: {norm_ref}")
        print(f"  HYP: {norm_hyp}\n")

    mean_wer = float(np.mean([item["wer"] for item in reports]))
    print("=" * 60)
    print(f"Evaluated {len(reports)} samples. Mean WER: {mean_wer:.3f}")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fout:
            json.dump({"mean_wer": mean_wer, "samples": reports}, fout, ensure_ascii=False, indent=2)
        print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
