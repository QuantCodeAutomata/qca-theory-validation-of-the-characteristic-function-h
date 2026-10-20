"""
Fixed-frequency characteristic-function (CF) half-spread estimator.

Estimator at frequency a:
  phi_hat(u) = (1/n) * sum cos(u * Y_t)
  d_hat(a)   = 1 - sqrt(phi_hat(2a)) / phi_hat(a)^2
  c_CF(a)    = (1/a) * arctan(d_hat(a)^{1/4})   [if admissible]

Admissibility:
  phi_hat(a) > 0  AND  0 <= phi_hat(2a) <= phi_hat(a)^4

This follows from the CF identification identity:
  1 - sqrt(phi(2a)) / phi(a)^2 = tan^4(ac)  [exact under Roll]
"""

import numpy as np

from src.roll import roll_estimator_batch


def cf_estimator_single(Y: np.ndarray, a: float) -> dict:
    """
    CF estimator at single frequency a for one path.

    Parameters
    ----------
    Y : np.ndarray of shape (n,)
    a : float, query frequency (must satisfy 0 < a*c < pi/4 for consistency)

    Returns
    -------
    dict with keys:
      c_hat      : float (NaN if inadmissible)
      admissible : bool
      phi_a      : float, phi_hat(a)
      phi_2a     : float, phi_hat(2a)
      d_hat      : float (NaN if inadmissible)
    """
    Y = np.asarray(Y, dtype=np.float64)
    phi_a  = float(np.mean(np.cos(a * Y)))
    phi_2a = float(np.mean(np.cos(2.0 * a * Y)))

    admissible = (phi_a > 0.0) and (phi_2a >= 0.0) and (phi_2a <= phi_a**4)

    if admissible:
        d_hat = float(np.clip(1.0 - np.sqrt(phi_2a) / phi_a**2, 0.0, 1.0))
        c_hat = float((1.0 / a) * np.arctan(d_hat**0.25))
    else:
        d_hat = float('nan')
        c_hat = float('nan')

    return {
        'c_hat': c_hat,
        'admissible': admissible,
        'phi_a': phi_a,
        'phi_2a': phi_2a,
        'd_hat': d_hat,
    }


def cf_estimator_batch(paths: np.ndarray, a: float) -> dict:
    """
    Apply CF estimator at frequency a to a batch of paths.

    Parameters
    ----------
    paths : np.ndarray of shape (n_paths, n)
    a     : float, query frequency

    Returns
    -------
    dict with keys:
      c_hat      : np.ndarray (NaN where inadmissible)
      admissible : np.ndarray of bool
      phi_a      : np.ndarray
      phi_2a     : np.ndarray
    """
    paths = np.asarray(paths, dtype=np.float64)
    phi_a  = np.mean(np.cos(a * paths), axis=1)
    phi_2a = np.mean(np.cos(2.0 * a * paths), axis=1)

    admissible = (phi_a > 0.0) & (phi_2a >= 0.0) & (phi_2a <= phi_a**4)

    # Safe sqrt: only compute where phi_2a >= 0
    safe_phi_2a = np.where(admissible, phi_2a, 0.0)
    safe_phi_a  = np.where(admissible, phi_a,  1.0)

    d_hat = np.clip(1.0 - np.sqrt(safe_phi_2a) / safe_phi_a**2, 0.0, 1.0)
    c_hat = np.where(admissible, (1.0 / a) * np.arctan(d_hat**0.25), np.nan)

    return {
        'c_hat': c_hat,
        'admissible': admissible,
        'phi_a': phi_a,
        'phi_2a': phi_2a,
    }


def cf_roll_mixture_estimator(
    paths: np.ndarray, a: float, w: float
) -> np.ndarray:
    """
    Convex CF-Roll mixture estimator:
      c_w = (1 - w) * c_R + w * c_CF(a)

    For inadmissible CF paths, falls back to Roll only.

    Parameters
    ----------
    paths : np.ndarray of shape (n_paths, n)
    a     : float, CF frequency
    w     : float, mixture weight in [0, 1]

    Returns
    -------
    np.ndarray of shape (n_paths,)
    """
    c_roll = roll_estimator_batch(paths)
    cf_res = cf_estimator_batch(paths, a)
    c_cf   = cf_res['c_hat']
    adm    = cf_res['admissible']

    return np.where(adm, (1.0 - w) * c_roll + w * c_cf, c_roll)
