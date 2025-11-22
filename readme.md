## Требования

python3.12

## Настройка окружения для квантизации

1. Создание venv
```
python3 -m venv .venv && source .venv/bin/activate && pip3 install -r requirements.txt
```

2. Инициализация git submodules
```
git submodule sync --recursive && \
    git submodule update --init --recursive ptq/vosk-tts ptq/wavlm-base-plus && \
    git -C ptq/vosk-tts fetch && \
    git -C ptq/vosk-tts checkout 5e2a103fbf26bdfe8ce8b79a0e33c6cd3c1c60e5
```

Полные инструкции и команды см. в `ptq/README.md`.


## Краткое описание проделанной работы

- **PTQ-эксперименты VITS2**: применили `torch.ao.quantization` с `QuantStub`, статическую per-tensor INT8-квантизацию, оборачивая только Conv/ConvT/Linear-слои (≈99.8 % из ~42 M параметров). Метрики: размер FP32 модели 167 MB против 43 MB у INT8, RTF на CPU (M3) 0.115 → 0.060, WER после Whisper-ASR 0.196 → 0.198 (без деградации).
- **Замеры скорости** (RTX 4070 Ti + Ryzen 7 7800X3D): на CPU RTF снижается с 0.047 до 0.025 при batch=1 и с 0.096 до 0.040 при batch=2, на GPU FP32 достигает 0.009/0.005/0.002/0.001 при batch=1/2/4/8. 100 итераций + прогрев.
- **Портирование**: протестированы два направления. ExecuTorch (pipeline `torch.export.export → executorch.exir.to_edge_transform_and_lower → to_executorch`) — упираемся в символические выражения и неподдерживаемые ATen-операторы, требуется заранее знать shape’ы. ONNX экспортируется с минимальными правками и работает через ONNX Runtime.
- **VoskTTSMobile**: Android-приложение на Kotlin, запускает оригинальные и INT8 ONNX-модели на CPU, показывает метрики синтеза (RTF, длительность, скорость токенизации). Позволяет быстро сравнить качество прямо на устройстве.


### RTF по батчу 
|              | FP32, CPU | INT8, CPU | FP32, GPU |
|--------------|----------:|----------:|----------:|
| batch_size=1 | 0.047     | 0.043     | 0.009     |
| batch_size=2 | 0.096     | 0.039     | 0.005     |
| batch_size=4 | 0.045     | 0.039     | 0.002     |
| batch_size=8 | 0.040     |           | 0.001     |
### Сравнение оригинальной и PTQ-модели
| Модель              | Размер модели | RTF (on CPU M3) | WER после ASR |
|---------------------|---------------|-----------------|---------------|
| оригинальная модель | 167 MB        | 0.115           | 0.196         |
| int8                | **43 MB**     | **0.06**        | 0.198         |
