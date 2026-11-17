"""
Split conformal prediction, built from scratch.

The CQR, Texas demand and returns experiments import these functions.
"""
import numpy as np


def compute_scores(y_true, y_pred):
    """Absolute residual score: s_i = |y_i - f(x_i)|"""
    return np.abs(np.asarray(y_true) - np.asarray(y_pred))


def conformal_quantile(calibration_scores, alpha):
    """q_hat = the ceil((n+1)(1-alpha))-th smallest calibration score."""
    scores = np.sort(np.asarray(calibration_scores))
    n = len(scores)
    if n == 0:
        raise ValueError("calibration_scores is empty")

    rank = int(np.ceil((n + 1) * (1 - alpha)))
    # rank > n means not enough calibration points for this alpha,
    # so the only valid interval is infinite
    if rank > n:
        return np.inf
    return scores[rank - 1]


def conformal_interval(y_pred, q_hat):
    """Returns (lower, upper) bounds: [f(x) - q_hat, f(x) + q_hat]"""
    y_pred = np.asarray(y_pred)
    return y_pred - q_hat, y_pred + q_hat


def evaluate_coverage(y_true, lower, upper):
    y_true = np.asarray(y_true)
    covered = (y_true >= lower) & (y_true <= upper)
    return covered.mean()


def naive_gaussian_interval(y_pred, residual_std, z=1.96):
    """Baseline that assumes Gaussian errors: prediction +/- z * std."""
    y_pred = np.asarray(y_pred)
    return y_pred - z * residual_std, y_pred + z * residual_std
