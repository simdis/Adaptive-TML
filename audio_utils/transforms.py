import librosa
import numpy as np
from matplotlib.pyplot import cm
import torch


class SpectrogramTransformLibrosa(torch.nn.Module):
    """
    Class SpectrogramTransformLibrosa.
    This class implements a PyTorch Transform that, given an acquired audio waveform,
    computes its spectrogram as in the librosa library.
    """
    def __init__(
            self, n_fft: int = 512, hop_length: int = 512, top_db: int = 80
    ) -> None:
        """
        Init function of SpectrogramTransformLibrosa.
        :param n_fft: the number of Fourier Transforms used in STFT.
        :param hop_length: the distance between two consecutive windows in STFT.
        :param top_db: the maximum decibel value.
        """
        super(SpectrogramTransformLibrosa, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.top_db = top_db

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Forward function of the transform. It converts the audio into a spectrogram.
        :param spectrogram: the input audio waveform of shape (1, num_samples).
        :return: a torch Tensor containing the spectrogram.
            Its shape is (1, 1 + n_fft // 2, 1 + num_samples // hop_length).
        """
        spectrogram = spectrogram.numpy()[0]  # Convert to numpy array and remove the first dimension
        spectrogram = np.abs(librosa.stft(spectrogram,
                                          n_fft=self.n_fft,
                                          hop_length=self.hop_length
                                          ))
        spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max, top_db=self.top_db)
        return torch.Tensor(spectrogram.reshape(1, *spectrogram.shape))


class SpectrogramToColormapTransform(torch.nn.Module):
    """
    Class SpectrogramToColormapTransform.
    This class implements a PyTorch Transform that, given a spectrogram, converts
    it to a colored image by means of a colormap (default Viridis).
    """
    def __init__(self, color_map: cm = cm.viridis, db: bool = True) -> None:
        """
        Init function of SpectrogramToColormapTransform.
        :param color_map: a reference to a colormap object. Default Viridis colormap of Pyplot.
        :param db: a boolean flag that switches between decibel and amplitude based spectrogram.
        """
        super(SpectrogramToColormapTransform, self).__init__()
        self._color_map = color_map
        self._db = db

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Forward function of the transform. It colors the spectrogram through the saved colormap.
        :param spectrogram: a torch Tensor containing the input spectrogram.
        :return: the colored spectrogram.
        """
        if self._db:
            # The input is a decibel based spectrogram.
            spec = spectrogram.detach().numpy()
        else:
            # The input is an amplitude based spectrogram.
            spec = spectrogram.detach().log2().numpy()
        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))
        spec = self._color_map(spec)[:, :, :, :3]
        spec = np.transpose(spec, (0, 3, 1, 2))
        # The first dimension (of size 1) is removed to maintain a 3D tensor.
        return torch.Tensor(spec[0])


def spectrogram_transforms(
        n_fft: int = 512, hop_length: int = 512, top_db: int = 80
) -> torch.nn.Sequential:
    """
    Function that defines a sequence of transforms from the input audio to a colored spectrogram.
    :param n_fft: the number of Fourier Transforms used in STFT.
    :param hop_length: the distance between two consecutive windows in STFT.
    :param top_db: the maximum decibel value.
    :return: a PyTorch Sequential object containing the transforms.
    """
    return torch.nn.Sequential(
        # torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length),
        # torchaudio.transforms.AmplitudeToDB(stype='amplitude', top_db=top_db),
        SpectrogramTransformLibrosa(n_fft=n_fft, hop_length=hop_length, top_db=top_db),
        SpectrogramToColormapTransform(),
    )
