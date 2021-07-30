import numpy as np

from condensing import condensing

from typing import Optional, Tuple


class CondensedNearestNeighbor:
    """
    Class that provides a (scikit-learn based) k nearest neighbor whose input representation is condensed.
    """
    def __init__(
        self, num_neighbors: int = 1, adaptive_k: bool = True,
        shuffle: bool = False, perm_idxs: Optional[np.ndarray] = None, _verbose: bool = False
    ) -> None:
        """
        Init method of CondensingInTimeNearestNeighbor.
        :param num_neighbors: the strictly positive number of neighbors of the Incremental kNN.
            This value is used when adaptive_k=False.
        :param adaptive_k: a boolean flag. If true, the number of neighbors is always equal to
            the ceil of the square root of the number of samples
        :param shuffle: a boolean flag. If true, the samples and their labels are shuffled prior to condensing.
        :param perm_idxs: if not None, the permutation indices used during the initial shuffling.
        :param _verbose: a boolean flag. If true, verbose logging is printed.
        """
        assert num_neighbors > 0, "The number of neighbors 'num_neighbors' must be strictly positive."
        super(CondensedNearestNeighbor, self).__init__()
        self._neighbors = int(num_neighbors)
        self._adaptive_neighbors = adaptive_k
        self._shuffle = shuffle
        self._perm_idxs = perm_idxs
        self._x = None
        self._y = None
        self._knn = None
        self._verbose = _verbose

    def _update_knn(self) -> None:
        """
        Internal method that updates the scikit-learn kNN.
        :return: Nothing.
        """
        if self._adaptive_neighbors:
            self._knn = condensing.create_knn(self._x, self._y)
        else:
            self._knn = condensing.create_knn(self._x, self._y, k_=self._neighbors)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the IncrementalTinyNearestNeighbor.
        :param x: a numpy array of shape (n_samples, n_features) representing the samples the kNN classifies among.
        :param y: a numpy array of shape (n_samples,) representing the samples' labels.
        :return: Nothing.
        """
        self._x, self._y = condensing.apply_condensing(
            x=x, y=y,
            shuffle=self._shuffle,
            perm_idxs=self._perm_idxs,
            _verbose=self._verbose
        )
        self._update_knn()

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit transform method. It fits the IncrementalTinyNearestNeigbor and returns the condensed representation.
        :param x: a numpy array of shape (n_samples, n_features) representing the samples the kNN classifies among.
        :param y: a numpy array of shape (n_samples,) representing the samples' labels.
        :return: a tuple (x, y) of the condensed representation of x and y.
        """
        self.fit(x, y)
        return self._x, self._y

    def fit_resample(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clone of fit_transform
        :param x: a numpy array of shape (n_samples, n_features) representing the samples the kNN classifies among.
        :param y: a numpy array of shape (n_samples,) representing the samples' labels.
        :return: a tuple (x, y) of the condensed representation of x and y.
        """
        return self.fit_transform(x, y)

    def transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method that returns the current representation of IncrementalTinyNearestNeigbor, i.e., its samples and labels.
        :return:
        """
        return self._x, self._y

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict on a new incoming sample(s).

        :param x: a numpy array of shape (n_samples, n_features)
        :return: a numpy array of shape (n_samples, ) with the prediction.
        """
        return self._knn.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict probabilities on a new incoming sample(s).

        :param x: a numpy array of shape (n_samples, n_features)
        :return: a numpy array of shape (n_samples, n_classes) with the probability of each class.
        """
        return self._knn.predict_proba(x)

    def get_k_neighbors_labels(self, x: np.ndarray) -> np.ndarray:
        """
        Return the labels of the k nearest neighbors.
        :param x: a numpy array of shape (num_samples, n_features)
        :return: a numpy array of shape (k, ) containing the labels of the k nearest neighbors.
        """
        _, idxs = self._knn.kneighbors(x)
        return self._y[idxs]
