TTS по телефону

Участники:
* Степановский Михаил
* Аржа Антон
* Колесников Егор


## Настройка окружения для квантизации

1. Создание venv
```
python3 -m venv .venv && source .venv/bin/activate && pip3 install -r requirements.txt
```

2. Клонирование vits и билд monotonic_align
```
git clone https://github.com/alphacep/vosk-tts && \
    cd vosk-tts/training/vits2/monotonic_align && \
    python3 setup.py build_ext --inplace && cd ../../../../ 
```

3. Загрузка чекпоинта

```
./.venv/bin/hf download alphacep/vosk-tts-ru-multi G_1000.pth config.json --local-dir pretrained
```

4. Загрузка словаря

```
wget https://alphacephei.com/vosk/models/vosk-model-tts-ru-0.7-multi.zip && \
    unzip vosk-model-tts-ru-0.7-multi.zip && \
    mkdir -p db && \
    cp vosk-model-tts-ru-0.7-multi/dictionary db/dictionary && \
    rm vosk-model-tts-ru-0.7-multi.zip
```

5. Clone SLM 

```
git clone https://huggingface.co/microsoft/wavlm-base-plus
```

6. Датасет для калибровки ([ссылка для скачивания](https://disk.yandex.ru/d/xCYR-zzQ0OJpQg))

```
tar -xf subset_natasha_1k.tar && rm subset_natasha_1k.tar
```

7. Препроцессинг датасета

```
python3 scripts/build_audiopaths_sid_texts.py
```