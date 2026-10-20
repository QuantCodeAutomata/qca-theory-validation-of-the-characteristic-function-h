"""
Roll (1984) half-spread estimator.

Estimator:
  lambda_R = -(1/(n-1)) * sum_{t=1}^{n-1} Y_t * Y_{t+1}
  c_R      = sqrt(max(0, lambda_R))
"""

import numpy as np


def roll_estimator(Y: np.ndarray) -> float:
    """
    Compute Roll estimator from a price-change series.

    Parameters
    ----------
    Y : np.ndarray of shape (n,), daily price changes

    Returns
    -------
    float, estimated half-spread (non-negative)
    """
    Y = np.asarray(Y, dtype=np.float64)
    n = len(Y)
    if n < 2:
        return float('nan')
    lambda_R = -float(np.mean(Y[:-1] * Y[1:]))
    return float(np.sqrt(max(0.0, lambda_R)))


def roll_estimator_batch(paths: np.ndarray) -> np.ndarray:
    """
    Apply Roll estimator to a batch of paths.

    Parameters
    ----------
    paths : np.ndarray of shape (n_paths, n)

    Returns
    -------
    np.ndarray of shape (n_paths,), estimated half-spreads
    """
    paths = np.asarray(paths, dtype=np.float64)
    lambda_R = -np.mean(paths[:, :-1] * paths[:, 1:], axis=1)
    return np.sqrt(np.maximum(0.0, lambda_R))
