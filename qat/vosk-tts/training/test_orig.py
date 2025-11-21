from vosk_tts.model import Model
from vosk_tts.synth import Synth
model_dir = "/home/michael/Documents/ITMO/EDLM/vosk-ft-test/vosk-tts/training/vosk-model-tts-ru-0.7-multi"

model = Model(model_path=model_dir, model_name=None, lang=None)

synth = Synth(model)

text = "Привет, это тест синтеза речи до дообучения модели."
out_wav = "before_ft_test.wav"

synth.synth(
    text=text,
    oname=out_wav,
    speaker_id=2,
    noise_level=None,
    speech_rate=1.0,
    duration_noise_level=None,
    scale=None,
)

print("Ready:", out_wav)