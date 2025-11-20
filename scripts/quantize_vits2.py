#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

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
import data_utils  # noqa: E402
from models import SynthesizerTrn  # noqa: E402
from text.symbols import symbols  # noqa: E402

# PTQ helpers (from exp/egor/ptq.py)
from exp.egor.ptq import (  # noqa: E402
    quantize_ptq_convs_only,
    prepare_model_for_ptq_convs_only,
    convert_model_from_ptq,
)

logger = logging.getLogger("quantize_vits2")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class TextOnlyCalibrationDataset(Dataset):
    """
    Dataset that reuses TextAudioSpeakerLoader text normalization without loading audio.
    Emits (text_tensor, sid_tensor).
    """

    def __init__(self, filelist_path: str, hparams):
        # Reuse loader internals for text normalization settings
        self.base = data_utils.TextAudioSpeakerLoader(filelist_path, hparams)

    def __len__(self) -> int:
        return len(self.base.audiopaths_sid_text)

    def __getitem__(self, idx) -> Tuple[torch.LongTensor, torch.LongTensor]:
        # audiopath, sid, text, cleaned_text
        _, sid, text, cleaned_text = self.base.audiopaths_sid_text[idx]
        text_tensor = self.base.get_text(text, cleaned_text)
        sid_tensor = self.base.get_sid(sid)
        return text_tensor, sid_tensor


class TextSpeakerCollate:
    """
    Collate for (text_tensor, sid_tensor) -> (text_padded, text_lengths, sids)
    """

    def __call__(self, batch):
        texts, sids = zip(*batch)
        text_lengths = torch.LongTensor([t.size(0) for t in texts])
        max_len = int(text_lengths.max().item())

        text_padded = torch.zeros(len(texts), max_len, dtype=torch.long)
        for i, t in enumerate(texts):
            text_padded[i, : t.size(0)] = t

        sids = torch.stack(sids).long().view(-1)
        return text_padded, text_lengths, sids


def build_model(hps) -> SynthesizerTrn:
    # Determine posterior channels same way as onnx_export.py
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
        **vars(hps.model),
    ).to("cpu")
    return net_g


def main():
    """
    Simplified entry: hardcoded parameters matching the notebook.
    - config: pretrained/config.json
    - checkpoint: pretrained/G_1000.pth
    - filelist: natasha_dataset/audiopaths_sid_text.txt (repo-local)
    - backend: fbgemm
    - module_roots: None (search whole model)
    - calib_batches: 30
    - batch_size: 8
    - out: exp/egor/G_quantized_int8.pth
    """

    torch.set_num_threads(max(1, os.cpu_count() or 1))

    # Load hparams
    config_path = PROJECT_ROOT / "pretrained" / "config.json"
    hps = utils.get_hparams_from_file(str(config_path))
    # Ensure text processing matches notebook calibration approach
    hps.data.aligned_text = False
    hps.data.g2p_text = True

    # Build model and load float checkpoint
    net_g = build_model(hps)
    ckpt_path = PROJECT_ROOT / "pretrained" / "G_1000.pth"
    utils.load_checkpoint(str(ckpt_path), net_g, None)
    net_g.eval().to("cpu")

    # Remove weight norm (as commonly done before export/quant)
    try:
        net_g.dec.remove_weight_norm()
    except Exception:
        pass
    try:
        net_g.flow.remove_weight_norm()
    except Exception:
        pass

    # Calibration data
    repo_filelist = PROJECT_ROOT / "natasha_dataset" / "audiopaths_sid_text.txt"
    if not repo_filelist.exists():
        raise FileNotFoundError(
            f"Calibration file list not found at {repo_filelist}. "
            "Please generate it or adjust the path."
        )
    calib_dataset = TextOnlyCalibrationDataset(str(repo_filelist), hps.data)
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=TextSpeakerCollate(),
    )

    def calibration_fn(model: SynthesizerTrn) -> None:
        model.eval()
        with torch.inference_mode():
            for i, (x, x_lengths, sid) in enumerate(calib_loader):
                if i >= 30:
                    break
                x = x.to("cpu")
                x_lengths = x_lengths.to("cpu")
                sid = sid.to("cpu")
                _ = model.infer(
                    x=x,
                    x_lengths=x_lengths,
                    sid=sid,
                    noise_scale=0.667,
                    length_scale=1.0,
                    noise_scale_w=1.0,
                    max_len=None,
                )

    module_roots = None  # search the whole model
    backend = "fbgemm"  # as used in the notebook
    logger.info(f"Preparing PTQ wrappers (roots=WHOLE MODEL, backend={backend})")
    # End-to-end PTQ of Conv/ConvT/Linear across selected roots (or entire model)
    quantize_ptq_convs_only(
        net_g,
        calibration_fn=calibration_fn,
        module_roots=module_roots,
        backend=backend,
    )

    # Save state dict
    out_path = PROJECT_ROOT / "exp" / "egor" / "G_quantized_int8.pth"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net_g.state_dict()}, str(out_path))
    logger.info(f"Saved quantized checkpoint to: {out_path}")


if __name__ == "__main__":
    main()


