1. Скопировать сюда db/
2. В `vosk-tts/training/vits2/text/__init__.py` добавить `if symbol != "—" and symbol not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']`
3. `mkdir outputs`