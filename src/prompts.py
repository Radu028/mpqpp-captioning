LANGUAGES = {
    "da": "Danish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ro": "Romanian",
}

TEMPLATE = "<image_placeholder>\nBriefly describe the image in {}."


def caption_prompt(code):
    return TEMPLATE.format(LANGUAGES[code])
