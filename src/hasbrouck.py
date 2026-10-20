"""
Hasbrouck (2009) Bayesian MCMC half-spread estimator.

Reduced model (baseline, r_mt = 0):
  Y_t = c * (Q_t - Q_{t-1}) + U_t
  Q_t in {-1, +1} latent trade signs
  U_t ~ N(0, V)

Full model (optional):
  Y_t = c * (Q_t - Q_{t-1}) + beta_m * r_mt + U_t

Gibbs sampler steps:
  1. Sample Q_t | Q_{t-1}, Y_t, c, V  (sequential binary update)
  2. Sample c  | Q, Y, V              (conjugate Normal posterior)
  3. Sample V  | Q, Y, c              (conjugate InvGamma posterior)

Prior specification (Hasbrouck 2009 Section C, as reconstructed):
  c   ~ TruncatedNormal(mu=0.02, sigma^2=0.01, c>0)
       implemented as Normal(0.02, 0.01) + abs + clip
  V   ~ InvGamma(a0=3, b0=0.0001)

Note: exact prior values from the paper are not fully specified; the above
are a documented reconstruction.
"""

import time
import numpy as np


def hasbrouck_mcmc(
    Y: np.ndarray,
    r_m: np.ndarray = None,
    burn_in: int = 1000,
    n_draws: int = 1500,
    seed: int = None,
) -> dict:
    """
    Bayesian Gibbs sampler for half-spread c under Roll-style model.

    Parameters
    ----------
    Y       : np.ndarray of shape (n,), daily price changes
    r_m     : np.ndarray of shape (n,), market returns (default zeros)
    burn_in : int, number of burn-in iterations (paper: 1000)
    n_draws : int, number of retained draws (paper: 1500)
    seed    : int, optional random seed

    Returns
    -------
    dict with keys:
      c_hat   : float, posterior mean of c
      V_hat   : float, posterior mean of V
      c_draws : np.ndarray of shape (n_draws,)
      V_draws : np.ndarray of shape (n_draws,)
      runtime : float, elapsed seconds
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=np.float64)
    n = len(Y)

    if r_m is None:
        r_m = np.zeros(n, dtype=np.float64)
    else:
        r_m = np.asarray(r_m, dtype=np.float64)

    # Initialise c from Roll estimator to prevent degeneracy at c≈0.
    lambda_hat = -float(np.mean(Y[:-1] * Y[1:]))
    c = max(np.sqrt(max(0.0, lambda_hat)), 1e-4)
    # Roll-based V: Var(Y) = 2c^2 + V  →  V_init = Var(Y) - 2c^2
    V = max(float(np.var(Y)) - 2.0 * max(0.0, lambda_hat), 1e-8)

    # Initialise Q via a forward-filter pass with Roll-initialised (c, V).
    Q = np.ones(n + 1, dtype=np.float64)
    for t in range(n):
        q_prev = Q[t]
        mean_p = c * (1.0 - q_prev) + r_m[t]
        mean_m = c * (-1.0 - q_prev) + r_m[t]
        log_odds = (-0.5 / V) * ((Y[t] - mean_p)**2 - (Y[t] - mean_m)**2)
        prob_p = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -30.0, 30.0)))
        Q[t + 1] = 1.0 if rng.random() < prob_p else -1.0

    # Prior hyperparameters (Hasbrouck 2009 Section C reconstruction)
    mu0     = 0.02      # prior mean for c
    tau0_sq = 0.01      # prior variance for c
    a0      = 3.0       # InvGamma shape for V
    b0      = 0.0001    # InvGamma scale for V

    c_draws = np.zeros(n_draws)
    V_draws = np.zeros(n_draws)

    for it in range(burn_in + n_draws):
        # ------------------------------------------------------------------
        # Step 1: Full-conditional Gibbs update for each Q[t].
        # Y[t] = c*(Q[t+1]-Q[t]) + r_m[t] + U[t], so Q[t] appears in:
        #   - Y[t-1] (as "current"  state, given Q[t-1]="previous" state)
        #   - Y[t]   (as "previous" state, given Q[t+1]="current"  state)
        # Using both observations is critical for correct mixing.
        # ------------------------------------------------------------------
        for t in range(n + 1):
            log_ratio = 0.0  # log[P(Q[t]=+1|rest) / P(Q[t]=-1|rest)]

            # Contribution from Y[t] = c*(Q[t+1]-Q[t]) + r_m[t] + U[t]
            # Q[t] acts as "previous" state here
            if t < n:
                q_next = Q[t + 1]
                mean_p = c * (q_next - 1.0) + r_m[t]   # E[Y[t]|Q[t]=+1]
                mean_m = c * (q_next + 1.0) + r_m[t]   # E[Y[t]|Q[t]=-1]
                log_ratio += (-0.5 / V) * ((Y[t] - mean_p)**2
                                           - (Y[t] - mean_m)**2)

            # Contribution from Y[t-1] = c*(Q[t]-Q[t-1]) + r_m[t-1] + U[t-1]
            # Q[t] acts as "current" state here
            if t > 0:
                q_prev = Q[t - 1]
                mean_p = c * (1.0 - q_prev) + r_m[t - 1]   # E[Y[t-1]|Q[t]=+1]
                mean_m = c * (-1.0 - q_prev) + r_m[t - 1]  # E[Y[t-1]|Q[t]=-1]
                log_ratio += (-0.5 / V) * ((Y[t - 1] - mean_p)**2
                                           - (Y[t - 1] - mean_m)**2)

            prob_plus = 1.0 / (1.0 + np.exp(-np.clip(log_ratio, -30.0, 30.0)))
            Q[t] = 1.0 if rng.random() < prob_plus else -1.0

        # ------------------------------------------------------------------
        # Step 2: Sample c | Q, Y, V   (conjugate Normal)
        # ------------------------------------------------------------------
        dQ = Q[1:] - Q[:-1]          # shape (n,)
        resid_no_c = Y - r_m         # Y_t - r_mt  (shape n)
        # Likelihood: resid_no_c = c*dQ + U_t, U_t ~ N(0,V)
        sum_dQ2   = float(np.sum(dQ**2))
        sum_dQ_y  = float(np.sum(dQ * resid_no_c))
        # Posterior: c | . ~ N(mu_n, tau_n^2)
        tau_n_sq = 1.0 / (sum_dQ2 / V + 1.0 / tau0_sq)
        mu_n     = tau_n_sq * (sum_dQ_y / V + mu0 / tau0_sq)
        c_sample = rng.normal(mu_n, np.sqrt(tau_n_sq))
        c = max(1e-7, abs(c_sample))   # enforce c > 0

        # ------------------------------------------------------------------
        # Step 3: Sample V | Q, Y, c   (conjugate InvGamma)
        # ------------------------------------------------------------------
        resid = Y - c * dQ - r_m
        a_n   = a0 + n / 2.0
        b_n   = b0 + 0.5 * float(np.sum(resid**2))
        # InvGamma(a_n, b_n): sample as 1/Gamma(a_n, 1/b_n)
        V = 1.0 / rng.gamma(a_n, 1.0 / b_n)
        V = max(V, 1e-12)

        if it >= burn_in:
            idx = it - burn_in
            c_draws[idx] = c
            V_draws[idx] = V

    runtime = time.perf_counter() - t0
    return {
        'c_hat':   float(np.mean(c_draws)),
        'V_hat':   float(np.mean(V_draws)),
        'c_draws': c_draws,
        'V_draws': V_draws,
        'runtime': runtime,
    }


def hasbrouck_batch(
    paths: np.ndarray,
    burn_in: int = 1000,
    n_draws: int = 1500,
    base_seed: int = 42,
) -> dict:
    """
    Apply Hasbrouck estimator to a batch of paths (sequential).

    Parameters
    ----------
    paths     : np.ndarray of shape (n_paths, n)
    burn_in   : int
    n_draws   : int
    base_seed : int, seed offset (path i uses seed base_seed+i)

    Returns
    -------
    dict with keys: c_hat (np.ndarray), runtimes (np.ndarray)
    """
    n_paths  = len(paths)
    c_hats   = np.zeros(n_paths)
    runtimes = np.zeros(n_paths)
    for i in range(n_paths):
        res = hasbrouck_mcmc(
            paths[i], burn_in=burn_in, n_draws=n_draws, seed=base_seed + i
        )
        c_hats[i]   = res['c_hat']
        runtimes[i] = res['runtime']
    return {'c_hat': c_hats, 'runtimes': runtimes}
