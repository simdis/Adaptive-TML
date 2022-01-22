from typing import Optional, Union, Tuple, Callable

import numpy as np
import torch.utils.data.dataset

import scipy.stats


class SyntheticMultivariateNormalGridDataset(torch.utils.data.dataset.Dataset):
    """
    Class SyntheticMultivariateNormalGridDataset.
    It is a Pytorch dataset that generates square of given size from a multivariate normal distribution, with
    the possibility to introduce a concept drift on both the mean and the covariance of each class and with a
    configurable magnitude and duration. The classes each generate square belongs to are randomly initialized with
    mean and cov in the provided intervals.
    """
    def __init__(self, grid_size: int, num_classes: int, dataset_size: int = 1000,
                 mean_scale: Union[float, np.ndarray] = 1.0, mean_min: Union[float, np.ndarray] = 1.0,
                 cov_scale: Union[float, np.ndarray] = 1.0, cov_min: Union[float, np.ndarray] = 0.5,
                 mean_change_magnitude: Union[float, np.ndarray] = 1.0, mean_change_duration: int = 1,
                 cov_change_magnitude: Union[float, np.ndarray] = 1.0, cov_change_duration: int = 1,
                 change_beginning: int = 500,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 random_generator: Optional[np.random.Generator] = None, seed: int = None) -> None:
        """
        The init method of Synthetic Multivariate Normal Grid Dataset with
        :param grid_size: The size of the square.
        :param num_classes: The number of classes each square can belong to.
        :param dataset_size: The number of generated samples.
        :param mean_scale: The width of the range in which the mean of each class can be. The range can be the same
            for all classes (i.e., a float value) or a value per each class (i.e., a numpy array).
        :param mean_min: The minimum value in which the mean of each class can be (see mean_scale for types).
        :param cov_scale: The width of the range in which the cov of each class can be (see mean_scale for types).
        :param cov_min: The minimum value in which the mean of each class can be (see mean_scale for types).
        :param mean_change_magnitude: The magnitude of change on the mean. It can be a float, a numpy array with one
            value per class or a numpy array with one value per each point of the grid and each class. The provided
            value(s) is/are gradually applied during all the duration of the drift.
        :param mean_change_duration: The duration of concept drift on the mean (>= 1).
        :param cov_change_magnitude: The magnitude of change on the cov (see mean_change_magnitude for types).
        :param cov_change_duration: The duration of concept drift on the cov (>= 1).
        :param change_beginning: The number of samples after which the change begins (>= 1). If this value is bigger
            than the dataset size, no concept drift will occur.
        :param transform: a function containing the transformation of the input.
        :param target_transform: a function containing the transformation of the output label.
        :param random_generator: The random generator used to provide random values.
        :param seed: If the random generator is not provided, the seed at which the random generator is initialized.
        """
        super(SyntheticMultivariateNormalGridDataset, self).__init__()
        self._grid_size = grid_size
        self._num_classes = num_classes
        self.transform = transform
        self.target_transform = target_transform
        # Create the random generator object if not provided to create always the same dataset.
        # If provided, the given seed is assigned to the random generator.
        self._seed = seed
        self._set_rg(random_generator)
        # Create the "classes" randomly.
        self._classes_mean = None
        self._classes_cov = None
        self._init_random_classes(mean_scale, mean_min, cov_scale, cov_min)
        self._create_data(
            mean_change_magnitude, mean_change_duration, cov_change_magnitude,
            cov_change_duration, change_beginning, dataset_size
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
        _y = int(self._rg.integers(low=0, high=self._num_classes))
        _x = self._rg.multivariate_normal(
            mean=self._classes_mean[:, _y],
            cov=self._classes_cov[:, :, _y],
            size=1
        )
        return _x.reshape((3, self._grid_size, self._grid_size)), _y

    def _create_data(self, mean_change_magnitude: Union[float, np.ndarray] = 1.0, mean_change_duration: int = 1,
                     cov_change_magnitude: Union[float, np.ndarray] = 1.0, cov_change_duration: int = 1,
                     change_beginning: int = 500, dataset_size: int = 1000) -> None:
        # assert 0 < change_beginning < dataset_size - max(mean_change_duration, cov_change_duration), \
        #     f"Invalid Change Beginning time"
        assert 0 < change_beginning, f"Invalid Change Beginning time"
        assert 0 < mean_change_duration, f"Invalid Mean Change Duration"
        assert 0 < cov_change_duration, f"Invalid Cov Change Duration"
        # Create samples before change
        self._data = [self._generate_one_sample() for _ in range(change_beginning)]
        # Create samples during drift
        mean_change_magnitude /= mean_change_duration  # The change is gradual during the concept drift.
        cov_change_magnitude /= cov_change_duration
        for _ in range(min(mean_change_duration, cov_change_duration)):
            self._data.append(self._generate_one_sample())
            self._shift_classes(mean_change_magnitude / mean_change_duration, cov_change_magnitude)
        extra_drift_samples = abs(mean_change_duration - cov_change_duration)
        for _ in range(extra_drift_samples):
            self._data.append(self._generate_one_sample())
            if mean_change_duration > cov_change_duration:
                self._shift_classes(mean_change_magnitude, 0)
            else:
                self._shift_classes(0, cov_change_magnitude)
        # Create samples after change
        after_change_samples = dataset_size - change_beginning - max(mean_change_duration, cov_change_duration)
        self._data += [self._generate_one_sample() for _ in range(after_change_samples)]

    def _set_rg(self, random_generator: Optional[np.random.Generator] = None) -> None:
        """
        Internal method that sets the random generator to the one provided as parameter or creates a new one.
        :param random_generator: the random generator object.
        :return: None
        """
        if random_generator is None:
            self._rg = np.random.default_rng(seed=self._seed)
        else:
            self._rg = random_generator

    def reset_rg(self, random_generator: Optional[np.random.Generator] = None) -> None:
        """
        Reset the random generator to either the default one (with the saved seed) or the one provided.
        :param random_generator: the random generator object.
        :return: None
        """
        self._set_rg(random_generator)

    def _init_random_classes(
            self, mean_scale: Union[float, np.ndarray] = 1.0, mean_min: Union[float, np.ndarray] = 1.0,
            std_scale: Union[float, np.ndarray] = 1.0, std_min: Union[float, np.ndarray] = 0.5,
            independent_dimensions: bool = False
    ) -> None:
        """
        Method that defines randomly the characteristics of the classes (mean and std of a normal multivariate
        distribution) given the range of possible values.
        :param mean_scale: The width of the uniform interval of possible values for the values of the mean.
            This parameter is either a float or an array of floats (one per each class).
        :param mean_min: The minimum of the uniform interval of possible values for the values of the mean.
            This parameter is either a float or an array of floats (one per each class).
        :param std_scale: The width of the uniform interval of possible values for the values of the std.
            This parameter is either a float or an array of floats (one per each class).
        :param std_min: The minimum of the uniform interval of possible values for the values of the std.
            This parameter is either a float or an array of floats (one per each class).
        :return: None
        """
        for _par, _name in zip(
                [mean_scale, mean_min, std_scale, std_min], ["mean_scale", "mean_min", "std_scale", "std_min"]
        ):
            assert (isinstance(_par, float) or
                    (isinstance(_par, np.ndarray) and np.size(_par) == self._num_classes)), \
                f"The {_name} parameter should be either float or an array of shape ({self._num_classes}, )."
        self._classes_mean = self._rg.random((3 * self._grid_size * self._grid_size, self._num_classes))
        self._classes_mean = self._classes_mean * mean_scale + mean_min
        self._classes_cov = \
            np.zeros((3 * self._grid_size * self._grid_size, 3 * self._grid_size * self._grid_size, self._num_classes))
        if independent_dimensions:
            diagonals = self._rg.random(self._classes_mean.shape)
            for zz, _ in enumerate(diagonals.transpose()):
                if isinstance(std_scale, float):
                    self._classes_cov[:, :, zz] = np.diag(diagonals[:, zz] * std_scale + std_min)
                else:
                    self._classes_cov[:, :, zz] = np.diag(diagonals[:, zz] * std_scale[zz] + std_min[zz])
        else:
            is_positive_definite = False
            while not is_positive_definite:
                # Create randomly the base matrix
                base_matrix = \
                    self._rg.random((3 * self._grid_size * self._grid_size, 3 * self._grid_size * self._grid_size))
                # Convert the matrix into a positive semidefinite matrix.
                base_matrix = 0.5 * (base_matrix.transpose() + base_matrix)
                base_matrix += \
                    3 * self._grid_size * self._grid_size * np.diag(np.ones(3 * self._grid_size * self._grid_size))
                is_positive_definite = np.all(np.linalg.eigvals(base_matrix) > 0)
            # Create the matrices from an Inverse Wishart distribution
            cov_matrices = scipy.stats.invwishart(
                df=self._grid_size * self._grid_size * 6,
                scale=base_matrix,
                seed=self._rg
            ).rvs(self._num_classes)
            # Now that the covariance matrices have been created, save them.
            for zz, _m in enumerate(cov_matrices):
                self._classes_cov[:, :, zz] = _m

    def _shift_classes(self, mean_shift: Union[float, np.ndarray], std_shift: Union[float, np.ndarray]):
        """
        Method that introduces a drift in the classes.
        :param mean_shift: The shift of the values of the mean.
            This parameter is either a float or an array of floats (one per each class or one per each value).
        :param std_shift: The shift of the values of the std.
            This parameter is either a float or an array of floats (one per each class or one per each value).
        :return: None
        """
        assert (isinstance(mean_shift, float) or
                (isinstance(mean_shift, np.ndarray) and
                 (np.size(mean_shift) == self._num_classes or mean_shift.shape == self._classes_mean.shape))), \
            f"The mean_shift parameter should be either float or an array of shape ({self._num_classes}, ) " \
            + f"or {self._classes_mean.shape}"
        assert (isinstance(std_shift, float) or
                (isinstance(std_shift, np.ndarray) and
                 (np.size(std_shift) == self._num_classes or std_shift.shape == self._classes_cov.shape))), \
            f"The std_shift parameter should be either float or an array of shape ({self._num_classes}, ) " \
            + f"or {self._classes_cov.shape}"
        self._classes_mean += mean_shift
        self._classes_cov += std_shift
