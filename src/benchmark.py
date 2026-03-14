import time
import pandas as pd
from tqdm import tqdm

from data_loader import SpeechDataset
from whisper_model import WhisperModel
from metrics import compute_wer
from noise import add_gaussian_noise


def benchmark(model_size="base", noisy=False):

    dataset = SpeechDataset(num_samples=100)

    model = WhisperModel(model_size)

    results = []

    for i in tqdm(range(len(dataset))):

        audio, sr, reference = dataset[i]

        if noisy:
            audio = add_gaussian_noise(audio)

        start = time.time()

        prediction = model.transcribe(audio)

        latency = time.time() - start

        error = compute_wer(reference, prediction)

        if i < 3:
            print("\nModel:", model_size)
            print("Reference:", reference)
            print("Prediction:", prediction)
            print("WER:", error)
            print("------")

        results.append({
            "model": model_size,
            "sample": i,
            "wer": error,
            "latency": latency
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    all_results = []

    for model_size in ["tiny", "base", "small"]:

        print(f"\nRunning CLEAN benchmark for {model_size}")
        df_clean = benchmark(model_size, noisy=False)
        df_clean["condition"] = "clean"

        print(f"\nRunning NOISY benchmark for {model_size}")
        df_noisy = benchmark(model_size, noisy=True)
        df_noisy["condition"] = "noisy"

        all_results.append(df_clean)
        all_results.append(df_noisy)

    final = pd.concat(all_results)

    final.to_csv("results.csv", index=False)

    summary = final.groupby(["model", "condition"])[["wer", "latency"]].mean()

    print("\nSummary Results:")
    print(summary)

    print("\nBenchmark finished!")
