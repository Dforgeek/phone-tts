from pathlib import Path
import sys
import argparse
import json
import torch
import torch.nn as nn

PROJECT_ROOT = Path("/Users/egorkolesnikv/Documents/ai_talent/edl/vosk").resolve()
VITS2_PATH = PROJECT_ROOT / "vosk-tts" / "training" / "vits2"
if str(VITS2_PATH) not in sys.path:
    sys.path.insert(0, str(VITS2_PATH))

import models
import utils
from text.symbols import symbols
import text
import commons


class EncoderWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, sid: torch.Tensor):
        if self.model.n_speakers > 0:
            g = self.model.emb_g(sid).unsqueeze(-1)
        else:
            g = None
        x, m_p, logs_p, x_mask = self.model.enc_p(tokens, lengths, g=g)
        return x, m_p, logs_p, x_mask


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


def main():
    parser = argparse.ArgumentParser(description="Export encoder-only to ONNX")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "pretrained" / "config.json"))
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "pretrained" / "G_1000.pth"))
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "pretrained" / "encoder.onnx"))
    parser.add_argument("--speaker", type=int, default=1)
    parser.add_argument("--text", type=str, default="привет")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    model = build_model(config).cpu().eval()
    utils.load_checkpoint(args.checkpoint, model, None)
    wrapper = EncoderWrapper(model).cpu().eval()

    tokens = get_text_tensor(args.text, config)
    tokens_b = tokens.unsqueeze(0)  # [1, T]
    lengths = torch.LongTensor([tokens.size(0)])  # [1]
    sid = torch.LongTensor([args.speaker])  # [1]

    input_names = ["tokens", "lengths", "sid"]
    output_names = ["x", "m_p", "logs_p", "x_mask"]
    dynamic_axes = {
        "tokens": {1: "T"},
        "lengths": {0: "N"},
        "sid": {0: "N"},
        "x": {2: "T_enc"},
        "m_p": {2: "T_enc"},
        "logs_p": {2: "T_enc"},
        "x_mask": {2: "T_enc"},
    }

    torch.onnx.export(
        wrapper,
        (tokens_b, lengths, sid),
        args.out,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported encoder ONNX to: {args.out}")


if __name__ == "__main__":
    main()




