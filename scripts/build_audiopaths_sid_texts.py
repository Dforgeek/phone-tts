import os

def build_audiopaths_sid_text_from_marks(
    marks_path: str,
    out_path: str,
    data_root: str = ".",
    sid: int = 0,
):
    """
    marks.txt: "wavs/000000.wav|ТЕКСТ"
    out_file: "full_path.wav|sid|text|cleaned_text"
    """
    with open(marks_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            wav_rel, text = line.split("|", maxsplit=1)
            wav_path = os.path.join(data_root, wav_rel)
            # cleaned_text здесь просто тот же text
            fout.write(f"{wav_path}|{sid}|{text}|{text}\n")

    print(f"Written filelist to {out_path}")


# пример вызова
MARKS_PATH = "natasha_dataset/marks.txt"
FILELIST_PATH = "natasha_dataset/audiopaths_sid_text.txt"
build_audiopaths_sid_text_from_marks(MARKS_PATH, FILELIST_PATH, data_root="natasha_dataset")
