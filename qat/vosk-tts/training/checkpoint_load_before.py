import models
import text
import utils
import data_utils
import json
import commons
import torch
import numpy as np
from scipy.io.wavfile import write
with open(r'pretrained/config.json', 'r') as f:
    config = json.load(f)
device = 'cpu'



from text.symbols import symbols
net_g = models.SynthesizerTrn(
    len(symbols),
    80,
    config['train']['segment_size'] // config['data']['hop_length'],
    n_speakers=config['data']['n_speakers'],
    mas_noise_scale_initial=0.01,
    noise_scale_delta=2e-6,
    **config['model']).cpu()
utils.load_checkpoint(r"db-finetune/out/G_650.pth",
                    net_g,
                    None)
net_g.eval()
txt = 'Я дообученная модель, загруженная из чекп+оинта. Привет, мир.'
out = 'congrats_ft_100_orig'
def get_text(txt, config):
    text_norm = text.text_to_sequence_g2p(txt)
    if config['data']['add_blank']:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    print(text_norm)
    return text_norm
#get_text(txt, config)
def vcss(out, inputstr, i):  # single
    device = torch.device("cpu")  # ВАЖНО: quantized модель = CPU
    net_g.to(device)
    net_g.eval()

    stn_tst = get_text(inputstr, config)

    speed = 1.0
    output_dir = r'outputs'
    sid = torch.LongTensor([i]).to(device)

    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)

        o, o_mb, *_ = net_g.infer(
            x_tst,
            x_tst_lengths,
            sid=sid,
            noise_scale=.667,
            noise_scale_w=0.8,
            length_scale=1 / speed,
        )

        audio = o[0, 0].cpu().numpy() * 32768.0  # vol scale

    write(rf'{output_dir}/{out}.wav', config['data']['sampling_rate'], audio.astype(np.int16))
    print(rf'{out}.wav Generated!')

vcss(out, txt, 1)