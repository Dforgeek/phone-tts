import torch
from torch import nn
from models import SynthesizerTrn
from text.symbols import symbols
import utils
from train_finetune_QAT_everything_2 import prepare_model_for_qat
from torch.ao.quantization import convert

class OnnxWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, text, text_lengths, scales, sid):
        noise_scale  = scales[0].item()
        length_scale = scales[1].item()
        noise_scale_w = scales[2].item()

        audio, *_ = self.model.infer(
            text,
            text_lengths,
            sid=sid,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )
        return audio  # [B, 1, T]

PATH_TO_CONFIG = "db-finetune/config.json"
PATH_TO_MODEL = "db-finetune/out/G_QAT_260.pth"
SPEAKER_ID = torch.LongTensor([0])
SCALE_CONFIG = torch.FloatTensor([0.667, 1.0, 0.8])
OPSET_VERSION = 15

hps = utils.get_hparams_from_file(PATH_TO_CONFIG)

if getattr(hps.model, "use_mel_posterior_encoder", False):
    print("Using mel posterior encoder for VITS2")
    posterior_channels = 80
    hps.data.use_mel_posterior_encoder = True
else:
    print("Using lin posterior encoder for VITS1")
    posterior_channels = hps.data.filter_length // 2 + 1
    hps.data.use_mel_posterior_encoder = False

net_g = SynthesizerTrn(
    len(symbols),
    posterior_channels,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    is_onnx=True,
    **hps.model,
).cpu()

net_g = prepare_model_for_qat(net_g)
net_g = convert(net_g.eval(), inplace=False)

ckpt = torch.load(PATH_TO_MODEL, map_location="cpu")
missing, unexpected = net_g.load_state_dict(ckpt["model"], strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

net_g.eval()
onnx_model = OnnxWrapper(net_g).eval()

num_symbols = net_g.n_vocab

dmy_text = torch.randint(low=0, high=num_symbols, size=(1, 50), dtype=torch.long)
dmy_text_length = torch.LongTensor([dmy_text.size(1)])
dummy_input = (dmy_text, dmy_text_length, SCALE_CONFIG, SPEAKER_ID)

torch.onnx.export(
    model=onnx_model,
    args=dummy_input,
    f="model_qat.onnx",
    opset_version=OPSET_VERSION,
    verbose=True,
    input_names=["input", "input_lengths", "scales", "sid"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "phonemes"},
        "input_lengths": {0: "batch_size"},
        "output": {0: "batch_size", 2: "time"},
    },
)
