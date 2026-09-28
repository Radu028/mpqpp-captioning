"""The original Centurio instruction, changing only the target language name."""
from pathlib import Path

TEMPLATE = "<image_placeholder>\nBriefly describe the image in {}."


def caption_prompt(path):
    language = Path(path).read_text(encoding="utf-8").strip()
    if not language or not language.isascii() or not language.isalpha():
        raise ValueError(f"{path}: the prompt language must be one English word, such as Romanian")
    return TEMPLATE.format(language)
