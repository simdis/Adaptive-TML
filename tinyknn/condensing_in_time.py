import numpy as np
from tinyknn.condensed_nearest_neighbor import CondensedNearestNeighbor

from typing import Optional


class CondensingInTimeNearestNeighbor(CondensedNearestNeighbor):
    """
    CondensingInTimeNearestNeighbor class.
    This class implements a tiny k-nearest neighbor classifier. In addition to the standard
    scikit-learn kNN implementation, this variant can passively learn over time by adding
    misclassified samples to kNN knowledge set.
    """

    def __init__(
            self, num_neighbors: int = 1, adaptive_k: bool = True,
            shuffle: bool = False, perm_idxs: Optional[np.ndarray] = None,
            max_samples: int = 500, _verbose: bool = False
    ) -> None:
        """
        Init method of CondensingInTimeNearestNeighbor.
        :param num_neighbors: the strictly positive number of neighbors of the Incremental kNN.
            This value is used when adaptive_k=False.
        :param adaptive_k: a boolean flag. If true, the number of neighbors is always equal to
            the ceil of the square root of the number of samples.
        :param shuffle: a boolean flag. If true, the samples and their labels are shuffled prior to condensing.
        :param perm_idxs: if not None, the permutation indices used during the initial shuffling.
        :param max_samples: the maximum number of samples the kNN can save.
        :param _verbose: a boolean flag. If true, verbose logging is printed.
        """
        # assert num_neighbors > 0, "The number of neighbors 'num_neighbors' must be strictly positive."
        assert max_samples > 0, "The number of samples 'max_samples' must be strictly positive."
        super(CondensingInTimeNearestNeighbor, self).__init__(
            num_neighbors=num_neighbors, adaptive_k=adaptive_k,
            shuffle=shuffle, perm_idxs=perm_idxs, _verbose=_verbose
        )
        self._max_samples = max_samples

    def _check_max_samples(self) -> None:
        """
        Internal method that removes the oldest samples if they overcome the window max length.
        :return: Nothing.
        """
        samples_to_remove = self._get_num_samples() - self._max_samples
        if samples_to_remove > 0:
            # Remove the first samples_to_remove samples
            self._x = self._x[samples_to_remove:, :]
            self._y = self._y[samples_to_remove:]

    def predict(self, x: np.ndarray, y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict on a new incoming sample.

        WARNING. This method actually does not allow multiple predictions!
        :param x: a numpy array of shape (1, n_features)
        :param y_true: (optional to maintain compatibility w.r.t. scikit-learn).
        The supervised information of shape (1, ). It provided, allows a passive update of the kNN in case
        of wrong predictions.
        :return: a numpy array of shape (1, ) with the prediction.
        """
        assert len(x.shape) == 2 and x.shape[0] == 1, "The predict method allows for one prediction at a time."
        assert y_true is None or np.size(y_true) == 1, "The predict method allows for one prediction at a time."
        # Predict y
        y = self._knn.predict(x)
        # If the supervised information is provided, do a passive update.
        if y_true is not None and y_true != y:
            # Add the supervised information
            self._x = np.concatenate((self._x, x), axis=0)
            self._y = np.concatenate((self._y, y), axis=0)
            # Check the window length
            self._check_max_samples()
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
        return super().predict_proba(x)
