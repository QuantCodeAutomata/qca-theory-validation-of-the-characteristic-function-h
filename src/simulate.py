"""
Roll model simulation utilities.

Model:  Y_t = c*(Q_t - Q_{t-1}) + U_t
  Q_t  ~ iid Uniform{-1, +1}
  U_t  ~ iid N(0, V)
  Q, U independent

Theoretical moments:
  E[Y_t^2]       = 2*c^2 + V
  E[Y_t*Y_{t+1}] = -c^2
"""

import numpy as np


def simulate_roll_path(
    c: float, V: float, n: int = 252, rng=None
) -> np.ndarray:
    """
    Simulate one path of n daily price changes under Roll's model.

    Parameters
    ----------
    c   : float, half-spread (c > 0)
    V   : float, noise variance (V >= 0)
    n   : int, number of observations
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray of shape (n,)
    """
    if rng is None:
        rng = np.random.default_rng()
    Q = rng.choice(np.array([-1, 1]), size=n + 1).astype(np.float64)
    U = rng.normal(0.0, np.sqrt(max(V, 0.0)), size=n)
    return c * (Q[1:] - Q[:-1]) + U


def simulate_roll_paths(
    c: float, V: float,
    n_paths: int = 1000, n: int = 252,
    seed: int = 42
) -> np.ndarray:
    """
    Simulate n_paths independent Roll model paths.

    Parameters
    ----------
    c       : float, half-spread
    V       : float, noise variance
    n_paths : int, number of independent paths
    n       : int, length of each path
    seed    : int, random seed for reproducibility

    Returns
    -------
    np.ndarray of shape (n_paths, n)
    """
    rng = np.random.default_rng(seed)
    return np.stack(
        [simulate_roll_path(c, V, n, rng) for _ in range(n_paths)]
    )


def validate_moments(
    c: float, V: float, n_large: int = 100_000, seed: int = 0
) -> dict:
    """
    Empirically validate theoretical moment conditions with a long pilot run.

    Checks:
      E[Y_t^2]       ~ 2*c^2 + V
      E[Y_t*Y_{t+1}] ~ -c^2

    Parameters
    ----------
    c       : float, half-spread
    V       : float, noise variance
    n_large : int, pilot sample size
    seed    : int, random seed

    Returns
    -------
    dict with keys: E_Y2_theoretical, E_Y2_empirical,
                    E_YY1_theoretical, E_YY1_empirical
    """
    rng = np.random.default_rng(seed)
    Y = simulate_roll_path(c, V, n_large, rng)
    return {
        'E_Y2_theoretical':  2.0 * c**2 + V,
        'E_Y2_empirical':    float(np.mean(Y**2)),
        'E_YY1_theoretical': -c**2,
        'E_YY1_empirical':   float(np.mean(Y[:-1] * Y[1:])),
    }
