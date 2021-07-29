import numpy as np
from condensing import condensing

from typing import Optional


class IncrementalTinyNearestNeighbor:
    """
    IncrementalTinyNearestNeighbor class.
    This class implements an incremental tiny k-nearest neighbor classifier. In addition to the standard
    scikit-learn kNN implementation, this variant can incrementally learn over time.
    """
    def __init__(
            self, num_neighbors: int = 1, adaptive_k: bool = True,
            incremental_learning_active: bool = True
    ) -> None:
        """
        Init method of IncrementalTinyNearestNeighbor.
        :param num_neighbors: the strictly positive number of neighbors of the Incremental kNN.
            This value is used when adaptive_k=False.
        :param adaptive_k: a boolean flag. If true, the number of neighbors is always equal to
            the ceil of the square root of the number of samples
        :param incremental_learning_active: a boolean flag enabling or not the incremental learning at the beginning.
        """
        assert num_neighbors > 0, "The number of neighbors 'num_neighbors' must be strictly positive."
        super(IncrementalTinyNearestNeighbor, self).__init__()
        self._neighbors = int(num_neighbors)
        self._adaptive_neighbors = adaptive_k
        self._x = None
        self._y = None
        self._knn = None
        self._incremental = incremental_learning_active

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
        self._x = x
        self._y = y
        self._update_knn()

    def predict(self, x: np.ndarray, y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict on a new incoming sample(s).
        :param x: a numpy array of shape (n_samples, n_features)
        :param y_true: (optional to maintain compatibility w.r.t. scikit-learn).
        The supervised information of shape (n_samples, ). It is used only when the incremental learning is active.
        :return: a numpy array of shape (n_samples, ) with the predictions.
        """
        # Predict y
        y = self._knn.predict(x)
        # If the supervised information is provided, do an incremental step.
        if y_true is not None and self._incremental:
            # Add the supervised information
            self._x = np.concatenate((self._x, x), axis=0)
            self._y = np.concatenate((self._y, y), axis=0)
            self._update_knn()
        # Return prediction
        return y

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict probabilities on a new incoming sample(s).

        WARNING: this method does not implement incremental learning updates.
        :param x: a numpy array of shape (n_samples, n_features)
        :return: a numpy array of shape (n_samples, n_classes) with the probability of each class.
        """
        return self._knn.predict_proba(x)

    def enable_incremental_learning(self) -> None:
        """
        Switch on incremental learning.
        :return: Nothing.
        """
        self._incremental = True

    def disable_incremental_learning(self) -> None:
        """
        Switch off incremental learning.
        :return: Nothing.
        """
        self._incremental = False
