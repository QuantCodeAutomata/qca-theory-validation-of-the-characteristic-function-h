"""
Augmented ECF-GMM half-spread estimator.

Extends the empirical characteristic function GMM by appending the Roll
lag-one autocovariance moment E[Y_t * Y_{t+1}] = -c^2 to the ECF moments.

Model moments:
  f_j(c, V) = exp(-V*u_j^2/2) * cos^2(c*u_j)   for j=1,...,J  [ECF]
  f_{J+1}(c, V) = -c^2                                          [Roll]

Two-step feasible GMM:
  Step 1: identity weight W = I
  Step 2: W = inv(Omega_hat), long-run HAC covariance of Z_t

Note: frequency set u_1,...,u_J is a reconstruction choice because the paper
does not fully specify it. Default: [5.0, 10.0, 20.0, 40.0].
"""

import numpy as np
from scipy.optimize import minimize


# Default ECF frequencies (reconstruction -- paper does not fully specify)
DEFAULT_FREQS = np.array([5.0, 10.0, 20.0, 40.0])


def model_moments(c: float, V: float, freqs: np.ndarray) -> np.ndarray:
    """
    ECF model moments: E[cos(u_j * Y_t)] = exp(-V*u_j^2/2) * cos^2(c*u_j).

    Parameters
    ----------
    c, V  : Roll model parameters
    freqs : np.ndarray of shape (J,), query frequencies

    Returns
    -------
    np.ndarray of shape (J,)
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    return np.exp(-V * freqs**2 / 2.0) * np.cos(c * freqs)**2


def model_moments_augmented(
    c: float, V: float, freqs: np.ndarray
) -> np.ndarray:
    """
    Augmented model moments: [ECF moments, -c^2].

    The last element corresponds to E[Y_t * Y_{t+1}] = -c^2 under Roll.

    Parameters
    ----------
    c, V  : Roll model parameters
    freqs : np.ndarray of shape (J,)

    Returns
    -------
    np.ndarray of shape (J+1,)
    """
    m = model_moments(c, V, freqs)
    return np.append(m, -c**2)


def sample_moments(Y: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Sample ECF moments: (1/n) * sum_t cos(u_j * Y_t).

    Parameters
    ----------
    Y     : np.ndarray of shape (n,)
    freqs : np.ndarray of shape (J,)

    Returns
    -------
    np.ndarray of shape (J,)
    """
    Y = np.asarray(Y, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    return np.mean(np.cos(np.outer(Y, freqs)), axis=0)


def sample_moments_augmented(Y: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Augmented sample moments: [ECF moments, lag-one autocovariance].

    Parameters
    ----------
    Y     : np.ndarray of shape (n,)
    freqs : np.ndarray of shape (J,)

    Returns
    -------
    np.ndarray of shape (J+1,)
    """
    m = sample_moments(Y, freqs)
    lag_moment = float(np.mean(Y[:-1] * Y[1:]))
    return np.append(m, lag_moment)


def long_run_covariance(
    Y: np.ndarray, freqs: np.ndarray, n_lags: int = 2
) -> np.ndarray:
    """
    HAC long-run covariance of augmented moment vector Z_t:
      Z_t = (cos(u_1*Y_t), ..., cos(u_J*Y_t), Y_t*Y_{t+1})

    Omega = Gamma_0 + sum_{h=1}^{n_lags} (Gamma_h + Gamma_h')
    where Gamma_h = E[Z_t * Z_{t-h}'].

    Parameters
    ----------
    Y      : np.ndarray of shape (n,)
    freqs  : np.ndarray of shape (J,)
    n_lags : int, number of autocovariance lags (default 2, per paper)

    Returns
    -------
    np.ndarray of shape (J+1, J+1)
    """
    Y = np.asarray(Y, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    n = len(Y)

    # Build Z_t of length n-1 (aligned: cos uses Y[:-1], roll uses Y[:-1]*Y[1:])
    cos_mat  = np.cos(np.outer(Y[:-1], freqs))          # (n-1, J)
    roll_vec = (Y[:-1] * Y[1:]).reshape(-1, 1)           # (n-1, 1)
    Z = np.hstack([cos_mat, roll_vec])                    # (n-1, J+1)

    Z_c = Z - Z.mean(axis=0)
    T   = Z_c.shape[0]

    Omega = Z_c.T @ Z_c / T
    for h in range(1, n_lags + 1):
        if T > h:
            Gh = Z_c[h:].T @ Z_c[:-h] / T
            Omega = Omega + Gh + Gh.T
    return Omega


def _gmm_objective(
    params: np.ndarray,
    m_hat: np.ndarray,
    freqs: np.ndarray,
    W: np.ndarray
) -> float:
    """GMM criterion: (m_hat - m(theta))' W (m_hat - m(theta))."""
    c, V = float(params[0]), float(params[1])
    if c <= 0.0 or V <= 0.0:
        return 1e15
    m = model_moments_augmented(c, V, freqs)
    diff = m_hat - m
    return float(diff @ W @ diff)


def ecf_gmm_augmented(
    Y: np.ndarray,
    freqs: np.ndarray = DEFAULT_FREQS,
    n_starts: int = 5,
    seed: int = None
) -> dict:
    """
    Two-step feasible GMM with augmented ECF moments.

    Step 1 uses identity weight matrix.
    Step 2 uses HAC long-run covariance inverse as weight matrix.

    Parameters
    ----------
    Y        : np.ndarray of shape (n,), daily price changes
    freqs    : np.ndarray of shape (J,), ECF frequencies (default [5,10,20,40])
    n_starts : int, number of random restarts for optimizer
    seed     : int, optional seed for random restarts

    Returns
    -------
    dict with keys: c_hat, V_hat, converged
    """
    Y = np.asarray(Y, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    rng = np.random.default_rng(seed)

    m_hat = sample_moments_augmented(Y, freqs)
    J1 = len(m_hat)  # J + 1

    # ---- Step 1: identity weight ----
    W1 = np.eye(J1)
    best_c, best_V, best_val = 0.02, np.var(Y) / 4.0, np.inf

    for _ in range(n_starts):
        c0 = float(rng.uniform(0.001, 0.08))
        V0 = float(rng.uniform(1e-7, 0.02))
        try:
            res = minimize(
                _gmm_objective, [c0, V0],
                args=(m_hat, freqs, W1),
                method='L-BFGS-B',
                bounds=[(1e-6, 1.0), (1e-10, 1.0)],
                options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_val and res.x[0] > 0 and res.x[1] > 0:
                best_val = res.fun
                best_c, best_V = float(res.x[0]), float(res.x[1])
        except Exception:
            pass

    converged = False

    # ---- Step 2: feasible GMM with HAC weight ----
    try:
        Omega = long_run_covariance(Y, freqs)
        # Regularize for positive definiteness
        eig_min = np.linalg.eigvalsh(Omega).min()
        if eig_min < 1e-12:
            Omega += (1e-10 - eig_min + 1e-12) * np.eye(J1)
        W2 = np.linalg.inv(Omega)

        res2 = minimize(
            _gmm_objective, [best_c, best_V],
            args=(m_hat, freqs, W2),
            method='L-BFGS-B',
            bounds=[(1e-6, 1.0), (1e-10, 1.0)],
            options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if res2.x[0] > 0 and res2.x[1] > 0:
            best_c, best_V = float(res2.x[0]), float(res2.x[1])
            converged = res2.success
    except Exception:
        converged = False

    return {'c_hat': best_c, 'V_hat': best_V, 'converged': converged}


def ecf_gmm_augmented_batch(
    paths: np.ndarray,
    freqs: np.ndarray = DEFAULT_FREQS
) -> dict:
    """
    Apply augmented ECF-GMM to a batch of paths.

    Parameters
    ----------
    paths : np.ndarray of shape (n_paths, n)
    freqs : np.ndarray of shape (J,)

    Returns
    -------
    dict with keys: c_hat (np.ndarray), converged (np.ndarray of bool)
    """
    results = [ecf_gmm_augmented(paths[i], freqs) for i in range(len(paths))]
    c_hats    = np.array([r['c_hat']    for r in results])
    converged = np.array([r['converged'] for r in results])
    return {'c_hat': c_hats, 'converged': converged}
