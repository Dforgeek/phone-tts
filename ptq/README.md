## Настройка окружения для квантизации

1. Заходим в текущую директорию
```
cd ptq/ && mkdir output
```

1. Создание `.env`
```
echo "HOME=$(pwd)" >> ../.env && source ../.env
```

2. Сборка monotonic_align
```
cd vosk-tts/training/vits2/monotonic_align && \ 
    python3 setup.py build_ext --inplace && cd - >/dev/null 
```

4. Загрузка чекпоинта
```
./.venv/bin/hf download alphacep/vosk-tts-ru-multi G_1000.pth config.json --local-dir pretrained
```

5. Загрузка словаря
```
wget https://alphacephei.com/vosk/models/vosk-model-tts-ru-0.7-multi.zip && \
    unzip vosk-model-tts-ru-0.7-multi.zip && \
    mkdir -p db && \
    cp vosk-model-tts-ru-0.7-multi/dictionary db/dictionary && \
    rm vosk-model-tts-ru-0.7-multi.zip
```

6. Датасет для калибровки (ссылка для скачивания: https://disk.yandex.ru/d/xCYR-zzQ0OJpQg)
```
wget "https://drive.usercontent.google.com/download?id=1kJsRhc6pryGU7JYaKMiV_DDSK-gRuJko&export=download&authuser=0&confirm=t" -O subset_natasha_1k.tar && \
    tar -xf subset_natasha_1k.tar && rm subset_natasha_1k.tar
```

8. Препроцессинг датасета
```
python3 scripts/build_audiopaths_sid_texts.py
```

9. В  В `vosk-tts/training/vits2/text/__init__.py` добавить `if symbol != "—" and symbol not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']` в 52 строке.

## Скрипты (квантование, профилирование, экспорт ONNX)

#### Квантование PyTorch INT8 (Conv/ConvT/Linear по всей модели):
```
python scripts/quantize_vits2.py
```

#### Профилирование:
int8:
```
python scripts/profile_vits2.py \
  --config pretrained/config.json \
  --checkpoint models/G_quantized_int8.pth \
  --quantized --repeat 5 --save-wav output/congrats_q.wav
```
оригинальная модель
```
python scripts/profile_vits2.py \
  --config pretrained/config.json \
  --checkpoint pretrained/G_1000.pth \
  --repeat 5 --save-wav congrats_fp32.wav
```

#### Экспорт FP32 в ONNX:
int8:
```
python scripts/export_onnx_vits2.py \
  --config pretrained/config.json \
  --checkpoint models/G_quantized_int8.pth \
  --out models/model_int8.onnx \
  --quantized

```
оригинальнаая модель:
```
python scripts/export_onnx_vits2.py \
  --config pretrained/config.json \
  --checkpoint pretrained/G_1000.pth \
  --out models/model.onnx
```

#### Запуск ONNX-инференса:
int8:
```
python scripts/run_onnx_vits2.py \
  --onnx models/model_int8.onnx \
  --config pretrained/config.json \
  --text "С трев+ожным ч+увством бер+усь я з+а пер+о." \
  --speaker-id 0 \
  --out output/onnx_out_int8.wav
```

оригинальная модель
```
python scripts/run_onnx_vits2.py \
  --onnx models/model.onnx \
  --config pretrained/config.json \
  --text "С трев+ожным ч+увством бер+усь я з+а пер+о." \
  --speaker-id 0 \
  --out output/onnx_out.wav
```


