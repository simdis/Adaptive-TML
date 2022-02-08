import numpy as np
from condensing import condensing
from tinyknn import active_cdt_functions
from tinyknn.active_tiny_knn import AdaptiveNearestNeighbor

from typing import Any, Callable, Dict, Optional, Tuple


class AdaptiveHybridNearestNeighbor(AdaptiveNearestNeighbor):
    """
    AdaptiveHybridNearestNeighborClass.
    This class implements the Hybrid Tiny kNN algorithm classifier, that extends the
    Active Tiny kNN one by providing a passive update in addition to the active one.
    The passive update reflects the Condensing in Time algorithm, thus refers to this
    and the extended class for details.
    """
    def __init__(
        self, num_neighbors: int = 1, adaptive_k: bool = True,
        window_length: int = 100, cdt_threshold: float = 100,
        init_steps: int = 100, step_size: int = 1,
        use_condensing: bool = True, shuffle: bool = True,
        perm_idxs: Optional[np.ndarray] = None, _verbose: bool = False,
        cdt_metric: str = 'accuracy', adaptation_mode: str = 'fast', min_samples_to_restart: int = 5,
        cdt_init_fn: Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] =
        active_cdt_functions.initialize_cusum_cdt_accuracy,
        cdt_init_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Init method of AdaptiveHybridNearestNeighbor.
        :param num_neighbors: the strictly positive number of neighbors of the Incremental kNN.
            This value is used when adaptive_k=False.
        :param adaptive_k: a boolean flag. If true, the number of neighbors is always equal to
            the ceil of the square root of the number of samples.
        :param window_length: the strictly positive maximum size of the kNN knowledge set.
            The passive updates discard the oldest samples if this size is reached.
        :param cdt_threshold: the CDT threshold.
        :param init_steps: the number of steps used during CDT initialization, i.e., the number of
            samples that are initially used to initialize the CDT before the algorithm can really start.
        :param step_size: the number of (supervised) samples used by each CDT iteration.
        :param use_condensing: whether to use or not the condensing algorithm during the first kNN fit and
            during each adaptation.
        :param shuffle: a boolean flag. If true, the samples and their labels are shuffled prior to condensing.
        :param perm_idxs: if not None, the permutation indices used during the initial shuffling.
        :param _verbose: a boolean flag. If true, verbose logging is printed.
        :param cdt_metric: the metric the CDT uses. Actually, the options are:
            1) accuracy, i.e., the CDT monitors the kNN accuracy to inspect for changes.
            2) confidence, i.e., the CDT monitors the kNN classification confidence to inspect for changes.
        :param adaptation_mode: the way the CDT is initialized again after a change. Actually, the options are:
            1) fast, i.e., the CDT is initialized on the history window.
            2) window, i.e., the CDT is initialized on the following 'init_steps' samples.
        :param min_samples_to_restart: a strictly positive integer representing the minimum number of
            samples within the history window to consider a change.
        :param cdt_init_fn: the function that Active Tiny kNN uses to initialize the CDT.
        :param cdt_init_kwargs: the args of cdt_init_fn
        """
        super(AdaptiveHybridNearestNeighbor, self).__init__(
            num_neighbors=num_neighbors, adaptive_k=adaptive_k,
            history_window_length=window_length, cdt_threshold=cdt_threshold,
            init_steps=init_steps, step_size=step_size,
            use_condensing=use_condensing, shuffle=shuffle,
            perm_idxs=perm_idxs, _verbose=_verbose,
            cdt_metric=cdt_metric, adaptation_mode=adaptation_mode,
            min_samples_to_restart=min_samples_to_restart,
            cdt_init_fn=cdt_init_fn,
            cdt_init_kwargs=cdt_init_kwargs
        )
        # History window: add time information
        self._last_window_time = None

    def _update_knn(self) -> None:
        """
        Internal method that updates the scikit-learn kNN.
        :return: Nothing.
        """
        if self._adaptive_neighbors:
            self._knn = condensing.create_knn(self._last_window_x, self._last_window_y)
        else:
            self._knn = condensing.create_knn(
                self._last_window_x, self._last_window_y, k_=self._neighbors
            )

    def _adaptation(self, num_samples_after_jhat: int, jhat_absolute: int) -> None:
        """
        Internal method that carries out the adaptation to new working conditions.
        :param num_samples_after_jhat: the number of supervised samples after estimated
            change time.
        :jhat_absolute: the estimated change time.
        :return: Nothing.
        """
        filter_idx = self._last_window_time >= jhat_absolute
        # Use np.sum to count only the true values into filter_idx
        # Do not use np.size that counts all the samples available
        if np.sum(filter_idx) < self._min_samples_to_restart:
            # Ignore this change!
            # Update counter
            self._step_from_last_detection += 1
            if self._verbose:
                print('Change ignored due to samples less than {}'.format(
                    self._min_samples_to_restart)
                )
        else:
            # Continue adaptation.
            self._last_window_x = self._last_window_x[filter_idx]
            self._last_window_y = self._last_window_y[filter_idx]
            self._last_window_time = self._last_window_time[filter_idx]
            self._update_knn()
            # 3) Reset init? Or compute init values on such window?
            # Case 1 --> restart init (window adaptation mode)
            if self._adaptation_mode == 'window':
                self._is_initializing = True
                # Change initializing window to binomial size.
                self._init_steps = self._n
            else:
                # Case 2 --> restart CDT on changed_x and changed_y (fast adaptation mode)
                self._init_sequence = self._compute_cdt_stat(
                    x=self._last_window_x,
                    y_true=self._last_window_y
                )
                self._init_cdt()

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the AdaptiveHybridNearestNeighbor. After the fit, the CDT is in initialization mode.
        :param x: a numpy array of shape (n_samples, n_features) representing the samples the kNN classifies among.
        :param y: a numpy array of shape (n_samples,) representing the samples' labels.
        :return: Nothing.
        """
        super().fit(x, y)
        # Initialize the last window time
        self._last_window_time = np.zeros(self._last_window_y.shape)

    def predict(self, x: np.ndarray, y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict on a new incoming sample (and applies the CDT and the passive update
        if the supervised information is provided).

        WARNING. This method actually does not allow multiple predictions!
        :param x: a numpy array of shape (1, n_features)
        :param y_true: (optional to maintain compatibility w.r.t. scikit-learn).
        The supervised information of shape (1, ). It provided, allows a CDT step.
        :return: a numpy array of shape (1, ) with the prediction.
        """
        assert len(x.shape) == 2 and x.shape[0] == 1, "The predict method allows for one prediction at a time."
        assert y_true is None or np.size(y_true) == 1, "The predict method allows for one prediction at a time."
        # Predict y
        y = self._knn.predict(x)
        if y_true is not None:
            # Update counter (Warning: it counts only supervised samples!)
            self._current_step += 1
            # assert(self._cdt_metric != 'accuracy' or (self._cdt_metric == 'accuracy' and y_true is not None))

            # ## Active Update ###
            self._active_update(x, y, y_true)

            # ## Passive update ###
            if y != y_true:
                # Save new data (without exceeding the window length)
                self._last_window_x = np.concatenate((self._last_window_x, x), axis=0)[-self._window_length:]
                self._last_window_y = np.concatenate((self._last_window_y, y_true), axis=0)[-self._window_length:]
                self._last_window_time = np.append(self._last_window_time, self._current_step)[-self._window_length:]
                # Update kNN with all the samples
                self._update_knn()
        # Return prediction
        return y

    def get_knn_samples(self) -> int:
        """
        A function to get the number of samples within the kNN knowledge set.
        :return: an integer representing the number of samples in the kNN knowledge set.
        """
        return np.size(self._last_window_time)
