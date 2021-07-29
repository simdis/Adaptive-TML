import librosa
import torch

from typing import Optional


def spectrogram_loader_librosa(
        path: str, sample_rate: int = 22050, mono: bool = True, max_seconds: Optional[int] = None
) -> torch.Tensor:
    """
    A loader function that relies on the librosa library.
    :param path: the path to the audio file to load.
    :param sample_rate: the optional sample rate at which loading the audio file.
    :param mono: a boolean flag. If true, the signal is converted to mono.
    :param max_seconds: the maximum numbed of seconds to load in the audio file.
    :return: the loaded audio file as a bi-dimensional torch Tensor of shape (1, num_samples).
    """
    waveform, _ = librosa.load(path, sr=sample_rate, mono=mono, duration=max_seconds)
    return torch.Tensor(waveform.reshape((1, -1)))
