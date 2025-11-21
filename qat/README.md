## QAT-пайплайн (по шагам)

Инструкция, как запустить QAT на основе текущего содержимого 

### 0) Подготовка окружения и данных

- Перейти в папку `qat/vosk-tts/training`:

```bash
cd qat/vosk-tts/training
```

- Собрать `monotonic_align` (однократно):

```bash
cd monotonic_align
python3 setup.py build_ext --inplace
cd ..
```

- Загрузить чекпоинты
```
./.venv/bin/hf download alphacep/vosk-tts-ru-multi --local-dir pretrained
```

5. Загрузка словаря
```
wget https://alphacephei.com/vosk/models/vosk-model-tts-ru-0.7-multi.zip && \
    unzip vosk-model-tts-ru-0.7-multi.zip && \
    mkdir -p db && \
    cp vosk-model-tts-ru-0.7-multi/dictionary db/dictionary && \
    rm vosk-model-tts-ru-0.7-multi.zip
```


- Развернуть датасет тонкой настройки:

```bash
unzip db-finetune.zip  # создаст папку db-finetune/
```


### 1) Сначала сделать всё из README по финтюну — до запуска самого финтюна

Ориентируйтесь на `qat/vosk-tts/training/README.md` (раздел Finetuning), но запуск финтюна делаем на следующем шаге.


### 2) Запустить простой train_finetune.py (без QAT)

Скрипт уже настроен на `config_noQAT`:

```bash
python3 train_finetune.py
```

По умолчанию берётся конфиг `db-finetune/config_noQAT.json`, а чекпоинты и логи пишутся в `db-finetune/out` (см. строки в `train_finetune.py`).


### 3) После нескольких эпох — переименовать выходную папку

Когда получите несколько чекпоинтов и логи тензорборда в `db-finetune/out`, остановите обучение и переименуйте папку, чтобы не перезаписать результаты:

```bash
mv db-finetune/out db-finetune/out_ft_100
```

Название можно выбрать любое осмысленное (например, по количеству шагов/эпох).


### 4) Запустить QAT-дообучение

Откройте `train_finetune_QAT_everything_2.py` и поменяйте пути к последним чекпоинтам от шага 3 (район ~290 строки):

- Пример (подставьте свои имена файлов):
  - `db-finetune/out_ft_100/G_1250.pth`
  - `db-finetune/out_ft_100/D_1250.pth`
  - `db-finetune/out_ft_100/DUR_1250.pth`
  - при наличии WavLM-дискриминатора: `db-finetune/out_ft_100/WD_1250.pth`

Затем запустите:

```bash
python3 train_finetune_QAT_everything_2.py
```

Скрипт дообучит модель в QAT-режиме, сохраняя новые чекпоинты и логи.


### 5) Быстрая проверка чекпоинтов: генерация аудио

- Для не-квантованного (до QAT) варианта используйте:

```bash
python3 checkpoint_load_before.py
```

Он загрузит чекпоинт float-модели и сгенерирует пример в `outputs/`.

- Для квантованного (INT8) варианта используйте:

```bash
python3 checkpoint_load.py
```

Этот скрипт включает QAT-конфигурацию декодера, делает `convert(...)` и загружает INT8-чекпоинт, после чего кладёт аудио в `outputs/`.

При необходимости поправьте внутри скриптов пути к нужным чекпоинтам (`db-finetune/out*/*.pth`) и имя выходного файла.


### 6) Экспорт в ONNX после QAT

Экспорт:

```bash
python3 onnx_export_qat.py
```

Скрипт успешно формирует ONNX, но при попытке подменить этот ONNX в готовой модели и прогнать инференс по инструкции из обычного README может появиться ошибка:

- `[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid input name: scales`

Это известная проблема текущего конвейера QAT-экспорта: модель с Q/DQ может требовать дополнительной пост-обработки графа (или соответствия имен/входов), прежде чем её можно будет прозрачно подменять в прод-сборке. Если столкнулись с этим, зафиксируйте используемые версии ORT/torch/onnx и приложите получившийся модельный файл для дальнейшего разбирательства.


### Где что лежит

- Скрипты обучения и экспорта: `qat/vosk-tts/training/*.py`
- Конфиги:
  - базовый финтюн без QAT: `db-finetune/config_noQAT.json`
  - остальные параметры модели: `pretrained/config.json`
- Чекпоинты:
  - финтюн без QAT: `db-finetune/out_ft_XXX/*.pth` (после переименования)
  - результаты QAT: новые чекпоинты согласно настройкам скрипта
- Примеры аудио: `outputs/*.wav`


### Быстрый чек-лист

1) Подготовить окружение и распаковать `db-finetune/`  
2) Запустить `python3 train_finetune.py` (конфиг `config_noQAT`)  
3) Переименовать `db-finetune/out` → `db-finetune/out_ft_100` (или своё имя)  
4) Обновить пути в `train_finetune_QAT_everything_2.py` и запустить его  
5) Прогнать `checkpoint_load_before.py` и `checkpoint_load.py` → получить wav в `outputs/`  
6) Экспортировать ONNX `python3 onnx_export_qat.py` (см. известную проблему с ORT "scales")


