"""
Root finding for bias-neutral frequency a* where b(a*; c, V) = 0.

Uses dense grid for sign-change detection followed by Brent's method for
high-precision root refinement (scipy.optimize.brentq).

Custom implementation -- Context7 confirmed no library equivalent.
"""

import numpy as np
from scipy.optimize import brentq

from src.theory import cf_bias_coeff


def build_frequency_grid(c: float, n_points: int = 5000) -> np.ndarray:
    """
    Build dense equally-spaced grid over the admissible domain (0, pi/(4c)).

    Endpoints are excluded using small numerical offsets to avoid
    division-by-zero and d->0 singularities.

    Parameters
    ----------
    c       : float, half-spread
    n_points: int, number of interior grid points

    Returns
    -------
    np.ndarray of shape (n_points,)
    """
    a_min = 1e-8 / c           # tiny offset from 0
    a_max = (1.0 - 1e-6) * np.pi / (4.0 * c)
    return np.linspace(a_min, a_max, n_points)


def detect_sign_changes(
    a_grid: np.ndarray, bias: np.ndarray
) -> list:
    """
    Find all consecutive pairs (a_i, a_{i+1}) where bias changes sign.

    NaN values are ignored (treated as non-bracketing).

    Parameters
    ----------
    a_grid : np.ndarray, frequency grid
    bias   : np.ndarray, evaluated bias coefficients

    Returns
    -------
    list of (a_lo, a_hi) bracket tuples
    """
    brackets = []
    for i in range(len(bias) - 1):
        if np.isfinite(bias[i]) and np.isfinite(bias[i + 1]):
            if bias[i] * bias[i + 1] < 0.0:
                brackets.append((float(a_grid[i]), float(a_grid[i + 1])))
    return brackets


def find_bias_neutral_frequency(
    c: float, V: float, n_grid: int = 5000
) -> dict:
    """
    Find all bias-neutral frequencies a* where b(a*; c, V) = 0.

    Algorithm:
      1. Build dense grid over admissible domain.
      2. Evaluate cf_bias_coeff at all grid points.
      3. Detect sign changes (bracketing intervals).
      4. Refine each bracket using Brent's method.

    Parameters
    ----------
    c      : float, half-spread
    V      : float, noise variance
    n_grid : int, number of grid points for sign-change detection

    Returns
    -------
    dict with keys:
      roots    : list of float, all bias-neutral frequencies found
      a_grid   : np.ndarray, frequency grid used
      bias     : np.ndarray, bias values on grid
      brackets : list of (a_lo, a_hi) intervals containing roots
    """
    a_grid = build_frequency_grid(c, n_grid)
    bias = cf_bias_coeff(a_grid, c, V)

    brackets = detect_sign_changes(a_grid, bias)
    roots = []
    for lo, hi in brackets:
        try:
            root = brentq(
                cf_bias_coeff, lo, hi,
                args=(c, V),
                xtol=1e-14, rtol=1e-12,
                maxiter=200
            )
            roots.append(float(root))
        except ValueError:
            pass

    return {
        'roots': roots,
        'a_grid': a_grid,
        'bias': bias,
        'brackets': brackets,
    }


def find_interior_root(c: float, V: float, n_grid: int = 5000) -> float:
    """
    Return the primary interior bias-neutral frequency a*.

    The paper emphasises the root closest to (but strictly inside) the
    upper admissible boundary pi/(4c). With the sign pattern b < 0 near
    0 and b > 0 near the boundary, this is the last (rightmost) root.

    Parameters
    ----------
    c      : float, half-spread
    V      : float, noise variance
    n_grid : int, grid resolution

    Returns
    -------
    float, the bias-neutral frequency a* (NaN if none found)
    """
    result = find_bias_neutral_frequency(c, V, n_grid)
    if not result['roots']:
        return float('nan')
    # Return the rightmost (largest) root -- interior bias-neutral a*
    return result['roots'][-1]
