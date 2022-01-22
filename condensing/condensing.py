import numpy as np
from sklearn import neighbors

from typing import Optional, Tuple


def create_knn(x: np.ndarray, y: np.ndarray, k_: Optional[int] = None) -> neighbors.KNeighborsClassifier:
    """
    An utility function that creates a kNN classifier with the given data.
    The classifier is created with the scikit learn library and the number of
    neighbors is the ceil of the square root of the number of samples within data,
    if a value is not provided.

    This function does not perform any assertion on data validity.
    :param x: a numpy array of shape (n_samples, n_features) representing the samples the kNN classifies among.
    :param y: a numpy array of shape (n_samples,) representing the samples' labels.
    :param k_: the number of neighbors (if None this values is computed automatically)
    :return: the kNN classifier fitted on the parameters.
    """
    k_ = int(min(np.ceil(np.sqrt(x.shape[0])), x.shape[0])) if k_ is None else k_
    return neighbors.KNeighborsClassifier(n_neighbors=k_, weights="distance").fit(x, y)


def apply_condensing(
        x: np.ndarray, y: np.ndarray, shuffle: bool,
        perm_idxs: Optional[np.ndarray] = None,
        random_generator: Optional[np.random.Generator] = None,
        _verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    An utility function that condenses the representation of kNN classifier.
    The condensing algorithm is that of (Hart, 1968) with a random starting point.
    The kNN number of neighbors at each iteration is always set to
    the ceil of the square root of the number of samples within data.

    References:
    Hart, Peter. "The condensed nearest neighbor rule (corresp.)."
        IEEE transactions on information theory 14.3 (1968): 515-516.

    :param x: a numpy array of shape (n_samples, n_features) representing the samples the kNN classifies among.
    :param y: a numpy array of shape (n_samples,) representing the samples' labels.
    :param shuffle: a boolean flag. If true, the values in (x,y) are shuffled prior to condensing.
    :param perm_idxs: if not None, the permutation indices used during the initial shuffling.
    :param random_generator: an instance of a numpy generator, if one wants to pass it.
    :param _verbose: a boolean flag. If true, verbose logging is printed.
    :return: a tuple (x, y) of the condensed representation of x and y.
    """
    n_samples = x.shape[0]
    # Shuffle
    if shuffle:
        if perm_idxs is None and random_generator:
            perm_idxs = random_generator.permutation(n_samples)
        elif perm_idxs is None:
            perm_idxs = np.random.permutation(n_samples)
        x = x[perm_idxs]
        y = y[perm_idxs]

    # Define keep and discard sets.
    keep_mask = np.zeros(n_samples).astype(np.bool)
    discard_mask = np.ones(n_samples).astype(np.bool)
    # At the first iteration keep contains only the first sample.
    keep_mask[0] = 1
    discard_mask[0] = 0
    # Create initial kNN classifier.
    knn_ = create_knn(x[keep_mask], y[keep_mask])
    # Define some auxiliary variables.
    is_converged = False
    num_epochs = 0
    while not is_converged:
        are_masks_changed = False
        to_test = discard_mask.nonzero()[0]
        for ii in to_test:
            curr_pred = knn_.predict(x[ii].reshape(1, -1))
            # If the prediction is correct, nothing is done, otherwise the sample is moved to keep set.
            if not curr_pred == y[ii]:
                keep_mask[ii] = 1
                discard_mask[ii] = 0
                are_masks_changed = True
                # Recreate the kNN classifier.
                del knn_
                knn_ = create_knn(x[keep_mask], y[keep_mask])
        # At the end, check if there is no change w.r.t. the previous iteration.
        # In that case, the algorithm is over.
        if not are_masks_changed:
            is_converged = True
        num_epochs += 1
        if _verbose:
            print('Epoch {}: from {} to {} samples in keep (total {})'.format(
                num_epochs, n_samples - to_test.size, np.sum(keep_mask), n_samples))
    # Return condensed values.
    return x[keep_mask], y[keep_mask]