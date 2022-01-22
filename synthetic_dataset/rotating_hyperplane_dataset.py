from typing import Optional, Tuple, Callable

import numpy as np
import torch.utils.data.dataset

import skmultiflow.data


class RotatingHyperplaneGridDataset(torch.utils.data.dataset.Dataset):
    """
    Class RotatingHyperplaneGridDataset.
    It is a Pytorch 2-class dataset that is based on the well-known rotating hyperplane dataset, but generates
    squared data.
    """
    def __init__(self, grid_size: int, dataset_size: int = 1000,
                 n_drift_features: int = 4, mag_change: float = 0.0,
                 noise_percentage: float = 0.01, sigma_percentage: float = 0.01,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 seed: int = None) -> None:
        """
        The init method of Rotating Hyperplane Grid Dataset with 2 classes and
        :param grid_size: The size of the square (>= 2).
        :param dataset_size: The number of generated samples.
        :param n_drift_features: The number of features that drifts (2 <= n_drift_features <= 3 * grid_size ^ 2).
        :param mag_change: The magnitude of change for every example (from 0 to 1).
        :param noise_percentage: The percentage of noise in every sample (from 0 to 1).
        :param sigma_percentage: The probability of reverting the change direction (from 0 to 1).
        :param transform: a function containing the transformation of the input.
        :param target_transform: a function containing the transformation of the output label.
        :param seed: If the random generator is not provided, the seed at which the random generator is initialized.
        """
        super(RotatingHyperplaneGridDataset, self).__init__()
        self._grid_size = grid_size
        self.transform = transform
        self.target_transform = target_transform
        # Create the random generator object if not provided to create always the same dataset.
        # If provided, the given seed is assigned to the random generator.
        self._seed = seed
        # Create the data
        self._data_generator = skmultiflow.data.HyperplaneGenerator(
            n_features=3 * self._grid_size * self._grid_size,
            n_drift_features=n_drift_features,
            mag_change=mag_change,
            noise_percentage=noise_percentage,
            sigma_percentage=sigma_percentage
        )
        # Manually add the random generator to the data generator object.
        self._data_generator._random_state = self._seed
        self._data_generator._prepare_for_use()
        self._create_data(
            dataset_size
        )

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        sample, target = self._data[index]
        sample = torch.Tensor(sample)
        # target = torch.Tensor(target).long()
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self._data)

    def _generate_one_sample(self) -> Tuple[np.ndarray, int]:
        """
        Internal method that generates a new sample.
        :return: a tuple with the sample and its label
        """
        _x, _y = self._data_generator.next_sample()
        return _x.reshape((3, self._grid_size, self._grid_size)), int(_y)

    def _create_data(self, dataset_size: int = 1000) -> None:
        # Create samples before change
        self._data = [self._generate_one_sample() for _ in range(dataset_size)]
        # Normalize data
        data_mean = sum(x[0] for x in self._data) / len(self._data)
        data_std = np.sqrt(sum((x[0] - data_mean) ** 2 for x in self._data))
        self._data = [((x[0] - data_mean) / data_std, x[1]) for x in self._data]
