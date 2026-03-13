import whisper
import time
import pandas as pd
from jiwer import wer
from datasets import load_dataset

dataset = load_dataset("librispeech_asr", "clean", split="test[:100]")

def evaluate(model_size):

    model = whisper.load_model(model_size)

    results = []

    for sample in dataset:

        audio = sample["audio"]["array"]
        reference = sample["text"]

        start = time.time()

        prediction = model.transcribe(audio, fp16=False)["text"]

        latency = time.time() - start

        error = wer(reference, prediction)

        results.append({
            "model": model_size,
            "wer": error,
            "latency": latency
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    models = ["tiny", "base", "small"]

    all_results = []

    for m in models:
        df = evaluate(m)
        all_results.append(df)

    results = pd.concat(all_results)

    results.to_csv("results/wer_results.csv", index=False)
