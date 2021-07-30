import numpy as np
from typing import Callable, Tuple


def precompute_constant_binomial_to_normal_matrices(
        p0: float, q0: float, p1: np.ndarray, q1: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute Generalized Log-Likelihood Ratio matrices in the case of a Binomial Distribution B(p, n)
    approximated as a Normal one N(np, np(1-p)).
    :param p0: the value of p before change.
    :param q0: this value should be equal to 1 - p0
    :param p1: a numpy array containing a grid of values for p after the change
    :param q1: this value should be equal to 1 - p1
    :param n: the value of n
    :return: a tuple containing the four precomputed matrices.
    """
    alpha_0 = ((np.multiply(p1, q1) - (p0 * q0)) /
               (2 * (np.multiply(n * p0 * q0 * p1, q1))))

    beta_0 = ((q0 - q1) /
              (q0 * q1))

    gamma_0 = np.log(p0 * q0) - np.log(np.multiply(p1, q1))

    gamma_1 = ((n * (p0 * q1 - p1 * q0)) /
               (2 * q0 * q1))

    return alpha_0, beta_0, gamma_0, gamma_1


def precompute_constant_change_mean_normal_matrices(
        mu0: float, mu1: np.ndarray, sigma0: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute Generalized Log-Likelihood Ratio matrices in the case of a Normal distribution N(mu, sigma)
    with changing mean.
    :param mu0: the Normal mean before change.
    :param mu1: a numpy array containing a grid of values for mu after the change
    :param sigma0: the Normal value of sigma before change.
    :return: a tuple containing the four precomputed matrices.
    """
    alpha_0 = np.zeros(np.shape(mu1))  # Unused

    beta_0 = (mu1 - mu0) / sigma0

    gamma_0 = (mu0 * mu0 - np.square(mu1)) / (2 * sigma0)

    gamma_1 = np.zeros(0)  # Unused

    return alpha_0, beta_0, gamma_0, gamma_1


def compute_loglikelihood_ratio(
        alpha_0: np.ndarray, beta_0: np.ndarray, gamma_0: np.ndarray, gamma_1: np.ndarray, y_i: float
) -> np.ndarray:
    """
    Compute Generalized Log-Likelihood Ratio with the given (precomputed matrices).
    :param alpha_0: the matrix multiplying the squared realization.
    :param beta_0: the matrix multiplying the realization.
    :param gamma_0: a constant matrix.
    :param gamma_1: a second constant matrix.
    :param y_i: the current realization the log-likelihood ratio is computed for.
    :return: a two-dimensional numpy array containing the log-likelihood ratio for the given realization.
    """
    lgi = alpha_0 * y_i ** 2 + beta_0 * y_i + gamma_0 + gamma_1
    return lgi.reshape((1, -1))


def update_Sjk_matrix(
        sjk: np.ndarray, alpha_0: np.ndarray, beta_0: np.ndarray,
        gamma_0: np.ndarray, gamma_1: np.ndarray, y_i: float,
        log_likelihood_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]
) -> np.ndarray:
    """
    Update the Generalized Log-Likelihood Ratio Matrix. That matrix, of shape (k, np.size(alpha_0)) contains
    in each cell (j, upsilon) the sum of log-likelihood ratios from sample j to the last one (k) for the given
    value after change in the given column (that this function does not know).
    :param sjk: the matrix at previous iteration (initially should be an empty matrix with size 0).
    :param alpha_0: the matrix multiplying the squared realization.
    :param beta_0: the matrix multiplying the realization.
    :param gamma_0: a constant matrix.
    :param gamma_1: a second constant matrix.
    :param y_i: the current realization the log-likelihood ratio is computed for.
    :param log_likelihood_fn: the function that computes the log-likelihood.
    :return: the updated sjk matrix.
    """
    # Compute the log ratio at the current time.
    logratio_i = log_likelihood_fn(alpha_0, beta_0, gamma_0, gamma_1, y_i)
    # Check if the matrix is empty (first batch)
    if not np.size(sjk):
        sjk = logratio_i
    else:
        # Sum the current log ratio to the matrix Sjk in order to have already
        # the correct formulation Sjk = sum of log ratios from time j to k
        sjk += logratio_i
        # Append the newest log ratio
        sjk = np.concatenate((sjk, logratio_i))
    return sjk


def initialize_cusum_cdt_accuracy(
        init_sequence: np.ndarray, allow_above_p0: bool = True,
        n: int = 50, step: float = 0.01, tol: float = 0.03, eps: float = 1e-3, _verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize the CUSUM CDT inspecting changes in accuracy. The function defines a grid of possible accuracies
    after a change and initialize the CDT accordingly.
    :param init_sequence: a numpy array containing the initial sequence of accuracies.
    :param allow_above_p0: a boolean flag. If True, accuracies greater than the estimated current one are
        admissible for after-change values.
    :param n: the Binomial parameter n.
    :param step: the step between two consecutive possible accuracies after the change.
    :param tol: the minimum interval between the estimated current accuracy and any possible value after change.
    :param eps: a numerical stability coefficient.
    :param _verbose: a boolean flag. If true, verbose logging is printed.
    :return: the precomputed log-likelihood matrices.
    """
    p0 = np.mean(init_sequence)
    # Fix extreme cases, i.e. p0=0 and p0=1
    if np.sum(init_sequence) == 0:
        p0 = eps  # Add eps to avoid numerical problems
    elif np.sum(init_sequence) >= np.size(init_sequence):
        p0 = 1 - eps  # Subtract eps to avoid numerical problems
    if _verbose:
        print('Classifier Accuracy: {:.03f}'.format(p0))
    q0 = 1 - p0
    if allow_above_p0:
        p1 = np.arange(0 + step / 2, 1, step)
    else:
        p1 = np.arange(0 + step / 2, p0, step)
    mask = (p1 <= p0 - tol) | (p1 >= p0 + tol)
    p1 = p1[mask]
    q1 = 1 - p1
    # Constant precomputed matrices
    return precompute_constant_binomial_to_normal_matrices(p0, q0, p1, q1, n)


def initialize_cusum_cdt_change_normal_mean(
        init_sequence: np.ndarray, allow_above_mu0: bool = True,
        step: float = 0.05, tol: float = 0.10, max_d: float = 1, eps: float = 1e-8, _verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize the CUSUM CDT inspecting changes in a Normal distribution mean. The function defines a grid of possible
    values for the mean after the change.
    :param init_sequence: a numpy array containing the initial sequence of realizations to estimate the mean.
    :param allow_above_mu0: a boolean flag. If True, means greater than the estimated current one are
        admissible for after-change values.
    :param step: the step between two consecutive possible mean values after the change.
    :param tol: the minimum interval between the estimated current mean and any possible value after change.
    :param max_d: the maximum distance between the estimated current mean and any possible value after change.
    :param eps: a numerical stability coefficient.
    :param _verbose: a boolean flag. If true, verbose logging is printed.
    :return: the precomputed log-likelihood matrices.
    """
    mu0 = float(np.mean(init_sequence))
    sigma0 = np.std(init_sequence) + eps
    if _verbose:
        print('Mu {:.3f} Sigma {:.3f}'.format(mu0, sigma0))
    if allow_above_mu0:
        mu1 = np.arange(mu0 - max_d, mu0 + max_d, step)
    else:
        mu1 = np.arange(mu0 - max_d, mu0, step)
    mask = (mu1 <= mu0 - tol) | (mu1 >= mu0 + tol)
    mu1 = mu1[mask]

    return precompute_constant_change_mean_normal_matrices(mu0, mu1, sigma0 ** 2)
