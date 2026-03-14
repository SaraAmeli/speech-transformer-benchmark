import numpy as np
from datasets import load_dataset


class SpeechDataset:

    def __init__(self, num_samples=100):
        self.dataset = load_dataset(
            "librispeech_asr",
            "clean",
            split="test"
        )

        self.dataset = self.dataset.select(range(num_samples))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        text = sample["text"]

        audio = np.array(audio).astype(np.float32)

        return audio, sr, text
