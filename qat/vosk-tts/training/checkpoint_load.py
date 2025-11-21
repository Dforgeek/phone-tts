import json
import torch
import numpy as np
from scipy.io.wavfile import write

import models, text, commons, utils
from text.symbols import symbols
from torch.ao.quantization import convert
from train_finetune import enable_qat_for_decoder

with open('pretrained/config.json', 'r') as f:
    config = json.load(f)

device = torch.device("cpu")

def build_synth(cfg):
    return models.SynthesizerTrn(
        len(symbols),
        80,
        cfg['train']['segment_size'] // cfg['data']['hop_length'],
        n_speakers=cfg['data']['n_speakers'],
        mas_noise_scale_initial=0.01,
        noise_scale_delta=2e-6,
        **cfg['model'],
    ).cpu()

net_g = build_synth(config)

# ВАЖНО: архитектура должна быть такая же, как у сохранённой qint8-модели:
# 1) включаем QAT
enable_qat_for_decoder(net_g, backend="fbgemm")
# 2) сразу конвертим декодер
net_g.dec = convert(net_g.dec, inplace=False)

# 3) грузим INT8-чекпоинт
ckpt = torch.load("db-finetune/out/G_QAT_100.pth", map_location="cpu")
net_g.load_state_dict(ckpt["model"])
net_g.to(device)
net_g.eval()

def get_text(txt, cfg):
    text_norm = text.text_to_sequence_g2p(txt)
    if cfg['data']['add_blank']:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)

def vcss(out_name, input_str, speaker_id: int = 0):
    stn_tst = get_text(input_str, config)
    speed = 1.0
    output_dir = "outputs"

    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        sid = torch.LongTensor([speaker_id]).to(device)

        o, o_mb, *_ = net_g.infer(
            x_tst,
            x_tst_lengths,
            sid=sid,
            noise_scale=.667,
            noise_scale_w=0.8,
            length_scale=1 / speed,
        )
        audio = o[0, 0].cpu().numpy() * 32768.0

    write(f"{output_dir}/{out_name}.wav",
          config['data']['sampling_rate'],
          audio.astype(np.int16))
    print(f"{out_name}.wav Generated!")

if __name__ == "__main__":
    txt = "Я текст, сгенерированный квантованной моделью."
    vcss("congrats_int8", txt, 1)
