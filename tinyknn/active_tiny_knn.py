import numpy as np
from condensing import condensing
from tinyknn import active_cdt_functions

from typing import Any, Callable, Dict, Optional, Tuple


class AdaptiveNearestNeighbor:
    """
    AdaptiveNearestNeighbor class.
    This class implements the Active Tiny kNN algorithm classifier, that extends the
    scikit-learn implementation of a k-nearest neighbors classifier with an active
    change detection (and adaptation) module to keep the kNN knowledge set always
    updated with changes in the data generation process.
    The class is parametric in the kind of Change Detection Test it relies on to
    detect changes and in the adaptation modality, providing actually two different
    solutions for each of them.
    """
    def __init__(
        self, num_neighbors: int = 1, adaptive_k: bool = True,
        history_window_length: int = 100, cdt_threshold: float = 100, init_steps: int = 100, step_size: int = 1,
        use_condensing: bool = True, shuffle: bool = True,
        perm_idxs: Optional[np.ndarray] = None, _verbose: bool = False,
        cdt_metric: str = 'accuracy', adaptation_mode: str = 'fast', min_samples_to_restart: int = 5,
        cdt_init_fn: Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] =
        active_cdt_functions.initialize_cusum_cdt_accuracy,
        cdt_init_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Init method of AdaptiveNearestNeighbor.
        :param num_neighbors: the strictly positive number of neighbors of the Incremental kNN.
            This value is used when adaptive_k=False.
        :param adaptive_k: a boolean flag. If true, the number of neighbors is always equal to
            the ceil of the square root of the number of samples.
        :param history_window_length: the strictly positive size of the history window, that contains
            the last supervised information, then used during adaptation.
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
        :param cdt_init_kwargs: the args of cdt_init_fn.
        """
        assert num_neighbors > 0, "The number of neighbors 'num_neighbors' must be strictly positive."
        assert history_window_length > 0, "The history window size 'history_window_length' must be strictly positive."
        assert init_steps > 0, "The number of initial steps 'init_steps' must be strictly positive."
        assert step_size > 0, "The step size 'step_size' must be strictly positive."
        assert cdt_metric in ['accuracy', 'confidence'], "cdt_metric can be either 'accuracy' or 'confidence'"
        assert adaptation_mode in ['fast', 'window'], "adaptation_mode can be either 'fast' or 'window'"
        assert min_samples_to_restart > 0,\
            "The minimum number of samples to restart 'min_samples_to_restart' must be strictly positive."

        super(AdaptiveNearestNeighbor, self).__init__()
        self._neighbors = int(num_neighbors)
        self._adaptive_neighbors = adaptive_k
        self._shuffle = shuffle
        self._perm_idxs = perm_idxs
        self._knn = None
        # History window
        self._last_window_x = None
        self._last_window_y = None
        self._window_length = int(history_window_length)
        # CDT precomputed matrices
        self._alpha = None
        self._beta = None
        self._gamma_0 = None
        self._gamma_1 = None
        # CDT variables
        self._cdt_metric = cdt_metric
        self._cdt_init_fn = cdt_init_fn
        self._cdt_init_kwargs = cdt_init_kwargs if cdt_init_fn is not None else dict()
        self._cdt_sequence = np.zeros(step_size)
        self._cdt_sequence_idx = 0
        self._cdt_sequence_history = np.zeros(0)
        self._sjk = np.zeros(0)  # empty array
        self._gk = np.zeros(0)  # empty array
        self._h = cdt_threshold
        self._n = step_size
        # Initialization values
        self._init_steps = init_steps
        self._init_sequence = np.zeros(0)  # empty array
        self._is_initializing = False
        # Condensing flag
        self._use_condensing = use_condensing
        # Adaptation parameters
        self._adaptation_mode = adaptation_mode
        self._min_samples_to_restart = min_samples_to_restart
        # Verbose flag
        self._verbose = _verbose
        # Other fields, current step, history, etc.
        self._current_step = 0
        self._step_from_last_detection = 0
        self._detection_history = list()
        self._refined_history = list()
        self._num_samples = 0

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
        # Update num samples
        self._num_samples = np.size(self._last_window_y)

    def _active_update(self, x: np.ndarray, y: np.ndarray, y_true: np.ndarray) -> None:
        """
        Internal method that carries out an active update.
        :param x: a numpy array of shape (n_features, )
        :param y: the predicted label of x.
        :param y_true: the supervised information of shape (1, ).
        :return: Nothing.
        """
        # Compute CDT metric
        y_cdt = self._compute_cdt_stat(x, y=y, y_true=y_true)
        y_cdt = np.array(y_cdt).reshape((1,))
        # Continue initialization or perform a CDT step
        if self._is_initializing:
            # Add the cdt metric to init sequence
            self._init_sequence = np.concatenate((self._init_sequence, y_cdt)) \
                if np.size(self._init_sequence) else y_cdt
            # Check if init is complete
            if np.size(self._init_sequence) >= self._init_steps:
                self._init_cdt()
        else:
            self._update_cdt(y_cdt)

    def _init_cdt(self) -> None:
        """
        Internal method that initializes the CDT.
        :return: Nothing.
        """
        # Initialize CDT data structures
        if self._cdt_init_kwargs is not None:
            self._alpha, self._beta, self._gamma_0, self._gamma_1 = \
                self._cdt_init_fn(self._init_sequence, **self._cdt_init_kwargs)
        else:
            self._alpha, self._beta, self._gamma_0, self._gamma_1 = \
                self._cdt_init_fn(self._init_sequence)
        self._sjk = np.zeros(0)
        self._gk = np.zeros(0)
        # Stop initialization
        self._is_initializing = False
        self._init_sequence = np.zeros(0)  # reset with empty array
        self._step_from_last_detection = 0

    def _adaptation(self, num_samples_after_jhat: int, jhat_absolute: int) -> None:
        """
        Internal method that carries out the adaptation to new working conditions.
        :param num_samples_after_jhat: the number of supervised samples after estimated
            change time.
        :jhat_absolute: the estimated change time.
        :return: Nothing.
        """
        if num_samples_after_jhat < self._min_samples_to_restart:
            # Ignore this change!
            # Update counter
            self._step_from_last_detection += 1
            if self._verbose:
                print('Change ignored due to samples less than {}'.format(
                    self._min_samples_to_restart)
                )
        else:
            # Continue adaptation
            self._last_window_x = self._last_window_x[-num_samples_after_jhat:]
            self._last_window_y = self._last_window_y[-num_samples_after_jhat:]
            if self._use_condensing:
                # Apply condensing to the new data
                self._last_window_x, self._last_window_y = condensing.apply_condensing(
                    x=self._last_window_x,
                    y=self._last_window_y,
                    shuffle=False,  # Always False during intermediate updates.
                )
            self._update_knn()
            # 3) Reset init? Or compute init values on such window?
            # Case 1 --> restart init (window adaptation mode)
            if self._adaptation_mode == 'window':
                self._is_initializing = True
            else:
                # Case 2 --> restart CDT on changed_x and changed_y (fast adaptation mode)
                self._init_sequence = self._compute_cdt_stat(
                    x=self._last_window_x,
                    y_true=self._last_window_y
                )
                self._init_cdt()

    def _update_cdt(self, y_i: float) -> None:
        """
        Internal method that performs a CDT step, given the realization y_i.
        :param y_i: the last realization.
        :return: Nothing.
        """
        # Save y_i
        self._cdt_sequence[self._cdt_sequence_idx] = y_i
        self._cdt_sequence_idx += 1  # Increase index
        self._cdt_sequence_history = np.concatenate((self._cdt_sequence_history, y_i)) \
            if np.size(self._cdt_sequence_history) else np.array(y_i).reshape((1,))
        if self._cdt_sequence_idx == self._n:
            # self._step_from_last_detection += 1  # Increase the counter
            self._cdt_sequence_idx = 0  # Reset index
            # Compute CDT step: 1) update sjk matrix
            self._sjk = active_cdt_functions.update_Sjk_matrix(
                self._sjk, self._alpha, self._beta,
                self._gamma_0, self._gamma_1, np.sum(self._cdt_sequence),
                active_cdt_functions.compute_loglikelihood_ratio
            )
            # 2) Compute gk value
            new_gk = np.array(np.nanmax(self._sjk)).reshape((1,))
            self._gk = np.concatenate((self._gk, new_gk)) if np.size(self._gk) else new_gk
            # 3) Inspect for change
            if self._gk[self._step_from_last_detection] >= self._h:
                # CHANGE! Save detection time
                self._detection_history.append(self._current_step)
                # Adapt!
                # 1) Compute refined change time
                # Estimate jhat (the time is on rows, i.e., index 0)
                # The estimation is multiplied by the dimension of batches n!
                jhat = np.unravel_index(np.nanargmax(self._sjk), self._sjk.shape)[0] * self._n
                # 2) Use all the available samples from that time to learn the new kNN
                num_samples_after_jhat = (self._step_from_last_detection + 1) * self._n - jhat
                jhat_absolute = self._current_step - num_samples_after_jhat
                self._refined_history.append(jhat_absolute)
                if self._verbose:
                    print('Change detected at time {} (refined {})'.format(self._current_step, jhat_absolute))
                self._adaptation(num_samples_after_jhat, jhat_absolute)
            else:
                # Update counter
                self._step_from_last_detection += 1

    def _compute_cdt_stat(
            self, x: np.ndarray, y: Optional[np.ndarray] = None, y_true: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Internal function that computes the CDT metric.
        :param x: a numpy array representing the input.
        :param y: the predicted label(s).
        :param y_true: the supervised label(s).
        :return: Nothing.
        """
        if self._cdt_metric == 'accuracy':
            if y is None:
                return self._knn.predict(x) == y_true
            return y == y_true
        elif self._cdt_metric == 'confidence':
            # Now it assumes that since there one samples at a time we can reshape as 1, -1
            conf_scores = self._knn.predict_proba(x).reshape((1, -1))
            return np.max(conf_scores, axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the AdaptiveNearestNeighbor. After the fit, the CDT is in initialization mode.
        :param x: a numpy array of shape (n_samples, n_features) representing the samples the kNN classifies among.
        :param y: a numpy array of shape (n_samples,) representing the samples' labels.
        :return: Nothing.
        """
        if self._use_condensing:
            # First apply condensing
            x, y = condensing.apply_condensing(
                x=x, y=y, shuffle=self._shuffle,
                perm_idxs=self._perm_idxs
            )
        # Save the samples in x and y in the window.
        self._last_window_x = x
        self._last_window_y = y
        # Fit kNN
        self._update_knn()
        # Filter out the samples that cannot fit into the window
        self._last_window_x = x[-self._window_length:]
        self._last_window_y = y[-self._window_length:]
        # Start CDT Initialization
        self._current_step = 0
        self._is_initializing = True

    def predict(self, x: np.ndarray, y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict on a new incoming sample (and applies the CDT if the supervised information is provided).

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
            # Save new data (without exceeding the window length)
            self._last_window_x = np.concatenate((self._last_window_x, x), axis=0)[-self._window_length:]
            self._last_window_y = np.concatenate((self._last_window_y, y_true), axis=0)[-self._window_length:]
            # Active Update
            self._active_update(x, y, y_true)
        # Return prediction
        return y

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict probabilities on a new incoming sample(s).

        WARNING: this method does not implement active update. It should be used always before a predict.
        :param x: a numpy array of shape (n_samples, n_features)
        :return: a numpy array of shape (n_samples, n_classes) with the probability of each class.
        """
        return self._knn.predict_proba(x)

    def get_detections(self) -> np.ndarray:
        """
        A function to get the history of detections.
        :return: a numpy array containing all the time stamps of detections.
        """
        return np.array(self._detection_history, dtype=np.int32)

    def get_estimated_change_times(self) -> np.ndarray:
        """
        A function to get the history of estimated change times.
        :return: a numpy array containing all the time stamps of estimated change times.
        """
        return np.array(self._refined_history, dtype=np.int32)

    def get_cdt_metric_history(self) -> np.ndarray:
        """
        A function to get the history of CDT metric.
        :return: a numpy array containing the CDT sequence history.
        """
        return self._cdt_sequence_history

    def get_knn_samples(self) -> int:
        """
        A function to get the number of samples within the kNN knowledge set.
        :return: an integer representing the number of samples in the kNN knowledge set.
        """
        return self._num_samples
