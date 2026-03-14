import re
from jiwer import wer


def normalize(text):

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()

    return text


def compute_wer(reference, prediction):

    reference = normalize(reference)
    prediction = normalize(prediction)

    return wer(reference, prediction)
