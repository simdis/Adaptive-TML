import numpy as np
import torch
import torchaudio

import audio_utils.spectrogram_dataloader_pytorch

from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class SpectrogramNoisyFolder(audio_utils.spectrogram_dataloader_pytorch.SpectrogramFolder):
    def __init__(
            self,
            root: str,
            loader: Union[Callable[[str], Any], Callable[..., Any]],
            loader_kwargs: Optional[Dict[str, Any]] = None,
            extensions: Tuple[str, ...] = ('wav', ),
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            get_filenames: bool = False,
            class_names: Optional[List[str]] = None,
            noise_steps: int = 1,
            induce_reverb: bool = True,
            induce_speed_distortion: bool = True,
            speed_distortion: float = 0.8,
            induce_echo: bool = False,
            echo_gain_in: float = 0.7,
            echo_gain_out: float = 0.8,
            echo_delay: float = 80.0,
            echo_decay: float = 0.3
    ) -> None:
        """
        Init method of Spectrogram Folder.
        :param root: the path to the dataset.
        :param loader: the function used to read each sample.
        :param loader_kwargs: the arguments (optional) to the loader.
        :param extensions: a tuple containing all the valid extensions.
        :param transform: a function containing the transformation of the input.
        :param target_transform: a function containing the transformation of the output label.
        :param is_valid_file: a function that checks the validity of a file given its path.
            It should not be used if extensions is provided.
        :param get_filenames: a boolean flag. If true, the filename is provided along with the input and its label.
        :param class_names: the list of classes to be kept within the dataset folder.
        :param noise_steps: the number of steps to fully apply the noises (in the meanwhile
            the noise is gradually applied).
        :param induce_reverb: whether to introduce reverberation or not. In case of gradual
            noise introduction, the reverberation starts after one third samples.
        :param induce_speed_distortion: whether to introduce speed distortion in the sound.
            The distortion is intended as a change of sound speed (with the cut at the original length).
        :param speed_distortion: The relative change of speed (default 1.0).
        :param induce_echo: whether to apply or not echo to samples. In case of gradual
            noise introduction, the echo starts after two third samples. Moreover, three echos are added:
            one after the given delay, the second after twice such delay with 75% of the first echo's decay,
            the last one after 2.5 of such delay with 60% of first echo's decay
        :param echo_gain_in: echo gain_in parameter.
        :param echo_gain_out: echo gain_out parameter.
        :param echo_delay: echo delay parameter.
        :param echo_decay: echo decay parameter.
        """
        super(SpectrogramNoisyFolder, self).__init__(
            root=root, loader=loader, loader_kwargs=loader_kwargs,
            extensions=extensions, transform=transform, target_transform=target_transform,
            is_valid_file=is_valid_file, get_filenames=get_filenames, class_names=class_names
        )
        self._speed_distortion = speed_distortion
        self._gain_in = echo_gain_in
        self._gain_out = echo_gain_out
        self._echo_delay = echo_delay
        self._echo_decay = echo_decay
        self._noise_steps = noise_steps
        self._start_reverb = (self._noise_steps // 3)
        self._start_echo = 2 * self._start_reverb
        self._induce_reverb = induce_reverb
        self._induce_echo = induce_echo
        self._induce_noise = induce_speed_distortion
        self._effects = None
        self._distortion_step = (1 - self._speed_distortion) / self._noise_steps
        self._internal_calls = 0

    def __getitem__(self, index: int) -> Union[Tuple[Any, Any], Tuple[Any, Any, str]]:
        """
        Overrides get_item method.
        :param index: Index of sample.
        :return: a Tuple containing the sample, its label, and, if any, the filename.
        """
        # print(self._internal_calls)
        self._internal_calls += 1
        _effects = list()
        if self._induce_noise:
            new_speed = 1 - self._distortion_step * self._internal_calls \
                if self._internal_calls < self._noise_steps else self._speed_distortion
            _effects.append(["speed", f"{new_speed:.3f}"])
            _effects.append(["rate", f"{self.sample_rate}"])
        if self._induce_reverb and self._internal_calls > self._start_reverb:
            _effects.append(["reverb", "-w"])
        if self._induce_echo and self._internal_calls > self._start_echo:
            _effects.append(["echo", f"{self._gain_in}", f"{self._gain_out}",
                             f"{self._echo_delay}", f"{self._echo_decay}",
                             f"{self._echo_delay * 2}", f"{self._echo_decay * 0.75}",
                             f"{self._echo_delay * 2 + self._echo_delay // 2}", f"{self._echo_decay * 0.6}"])
        _effects.append(["remix", "-"])
        # Get data
        path, target = self.samples[index]
        sample = self.loader(path, **self.loader_kwargs)
        sample = torch.Tensor(sample)
        sample, _ = torchaudio.sox_effects.apply_effects_tensor(
            sample, self.sample_rate, _effects
        )
        sample = sample.view((1, -1))
        if sample.shape[1] < self.expected_dim:
            # Extend it by repeating
            factor = self.expected_dim // sample.shape[1] + 1
            sample = np.array(sample.numpy().ravel().tolist() * factor)
        else:
            # Check sample does not overcome the maximum size
            sample = sample.numpy()[:, :self.expected_dim]
        sample = torch.Tensor(sample.reshape((1, -1)))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.get_filenames:
            return sample, target, path
        return sample, target
