import numpy as np


def add_gaussian_noise(audio, noise_level=0.01):

    audio = audio.astype(np.float32)

    noise = np.random.normal(
        0,
        noise_level,
        audio.shape
    ).astype(np.float32)

    noisy_audio = audio + noise

    return noisy_audio
