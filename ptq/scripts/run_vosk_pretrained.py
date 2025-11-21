from pathlib import Path
import sys
import json
import torch
import numpy as np
from scipy.io.wavfile import write


VITS2_PATH = "vosk-tts/training/vits2"
if str(VITS2_PATH) not in sys.path:
    sys.path.insert(0, str(VITS2_PATH))

# Notebook imports
import models
import text
import utils
from text.symbols import symbols
import commons

# Load config used in the notebook
CONFIG_PATH = "pretrained/config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

device = "cpu"

# Build model as in the notebook
net_g = models.SynthesizerTrn(
    len(symbols),
    80,
    config["train"]["segment_size"] // config["data"]["hop_length"],
    n_speakers=config["data"]["n_speakers"],
    mas_noise_scale_initial=0.01,
    noise_scale_delta=2e-6,
    **config["model"],
).cpu()

# Load checkpoint
CHECKPOINT_PATH = "pretrained/G_1000.pth"
utils.load_checkpoint(str(CHECKPOINT_PATH), net_g, None)
net_g.eval()

# Text and output name (from notebook)
txt = "привет"
out = "congrats"


def get_text(input_text: str, cfg: dict) -> torch.LongTensor:
    text_norm = text.text_to_sequence_g2p(input_text)
    if cfg["data"]["add_blank"]:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm_tensor = torch.LongTensor(text_norm)
    print(text_norm_tensor)
    return text_norm_tensor


def vcss(out_name: str, input_text: str, speaker_id: int) -> None:
    stn_tst = get_text(input_text, config)
    speed = 1.0
    output_dir = Path(".")
    sid = torch.LongTensor([speaker_id]).to(device)
    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                sid=sid,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1 / speed,
            )[0][0, 0]
            .data.cpu()
            .numpy()
            * 32768.0
        )
        print(audio, np.max(audio))
    out_path = output_dir / f"{out_name}.wav"
    write(str(out_path), config["data"]["sampling_rate"], audio.astype(np.int16))
    print(f"{out_path} Generated!")


if __name__ == "__main__":
    vcss(out, txt, 1)


