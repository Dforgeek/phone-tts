#!/usr/bin/env python3
import argparse
import io
import os
from pathlib import Path
import tarfile

def main():
    p = argparse.ArgumentParser(description="Create a tar with the first N samples from marks.txt")
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Папка с marks.txt и подкаталогом wavs/")
    p.add_argument("--n", type=int, default=1000, help="Сколько первых строк взять из marks.txt")
    p.add_argument("--out", type=Path, required=True, help="Путь к выходному .tar")
    p.add_argument("--preserve-subdir", type=str, default=None,
                   help="Опционально: имя корневой папки внутри архива (например 'natasha_dataset'). "
                        "Если не задано — файлы будут лежать в корне архива.")
    args = p.parse_args()

    ds = args.dataset_root
    marks = ds / "marks.txt"
    wavs_dir = ds / "wavs"

    if not marks.is_file():
        raise FileNotFoundError(f"Не найден файл: {marks}")
    if not wavs_dir.is_dir():
        raise NotADirectoryError(f"Не найдена папка: {wavs_dir}")

    # читаем первые N строк
    with marks.open("r", encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            if i >= args.n:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                # пропускаем странные строки
                continue
            rel_wav, text = parts
            # нормализуем относительный путь
            rel_wav = rel_wav.strip()
            lines.append((rel_wav, text))

    if not lines:
        raise RuntimeError("Пустая выборка: не удалось прочитать ни одной валидной строки из marks.txt")

    # готовим новый marks.txt (только первые N)
    subset_marks_text = "\n".join(f"{rel}|{txt}" for rel, txt in lines) + "\n"

    # пакуем в TAR без предварительного копирования
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out, "w") as tar:
        # 1) добавим новый marks.txt как виртуальный файл
        marks_arcname = "marks.txt" if args.preserve_subdir is None else f"{args.preserve_subdir}/marks.txt"
        data = subset_marks_text.encode("utf-8")
        info = tarfile.TarInfo(name=marks_arcname)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

        # 2) добавим нужные wav'ы из исходной папки по их относительным путям
        for rel_wav, _ in lines:
            src = (ds / rel_wav).resolve()
            if not src.is_file():
                # Можно жёстче падать с ошибкой, но в продакшене лучше логировать и пропускать
                raise FileNotFoundError(f"В marks.txt указан файл, которого нет: {src}")
            # arcname — куда положим внутри архива
            if args.preserve_subdir is None:
                arcname = rel_wav
            else:
                arcname = f"{args.preserve_subdir}/{rel_wav}"
            # гарантируем, что подкаталоги есть (tar сам создаст структуру)
            tar.add(src, arcname=arcname)

    print(f"Готово: {args.out} (строк: {len(lines)})")

if __name__ == "__main__":
    main()
