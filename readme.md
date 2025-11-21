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


