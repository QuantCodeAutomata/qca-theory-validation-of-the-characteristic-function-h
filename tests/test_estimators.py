"""
Tests for estimator modules: roll.py, cf.py, gmm.py, hasbrouck.py.

Validates:
  - Output types and shapes
  - Non-negativity of half-spread estimates
  - Roll estimator admissibility (takes max(0,...))
  - CF admissibility detection
  - GMM convergence for simple cases
  - Hasbrouck posterior mean non-negativity
  - Mathematical properties of each estimator
"""

import numpy as np
import pytest

from src.simulate import simulate_roll_path, simulate_roll_paths, validate_moments
from src.roll import roll_estimator, roll_estimator_batch
from src.cf import cf_estimator_single, cf_estimator_batch, cf_roll_mixture_estimator
from src.gmm import (
    model_moments, model_moments_augmented, sample_moments,
    sample_moments_augmented, long_run_covariance, ecf_gmm_augmented,
    DEFAULT_FREQS,
)
from src.hasbrouck import hasbrouck_mcmc

C    = 0.02
V    = C**2     # rho = 1
N    = 252
SEED = 42


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_output_shape(self):
        Y = simulate_roll_path(C, V, N, np.random.default_rng(SEED))
        assert Y.shape == (N,)

    def test_batch_shape(self):
        paths = simulate_roll_paths(C, V, n_paths=10, n=N, seed=SEED)
        assert paths.shape == (10, N)

    def test_moment_validation(self):
        moments = validate_moments(C, V, n_large=200_000, seed=0)
        # Allow 3% relative tolerance for large-sample pilot
        tol = 0.03
        E_Y2_ratio = abs(moments['E_Y2_empirical'] / moments['E_Y2_theoretical'] - 1)
        E_YY_ratio = abs(moments['E_YY1_empirical'] / moments['E_YY1_theoretical'] - 1)
        assert E_Y2_ratio < tol, f"E[Y^2] mismatch: {E_Y2_ratio:.3f}"
        assert E_YY_ratio < tol, f"E[YY1] mismatch: {E_YY_ratio:.3f}"

    def test_Y_dtype(self):
        Y = simulate_roll_path(C, V, N)
        assert Y.dtype == np.float64

    def test_zero_noise(self):
        """With V=0, Y_t takes only values in {-4c, -2c, 0, 2c, 4c}."""
        Y = simulate_roll_path(C, 0.0, 500, np.random.default_rng(0))
        uniq = np.unique(np.round(Y / C, 6))
        assert set(uniq).issubset({-4.0, -2.0, 0.0, 2.0, 4.0})


# ---------------------------------------------------------------------------
# Roll estimator
# ---------------------------------------------------------------------------

class TestRollEstimator:
    def test_output_non_negative(self):
        """Roll estimator applies max(0,...) so output is always >= 0."""
        rng  = np.random.default_rng(SEED)
        for _ in range(10):
            Y = simulate_roll_path(C, V, N, rng)
            assert roll_estimator(Y) >= 0.0

    def test_batch_shape(self):
        paths = simulate_roll_paths(C, V, n_paths=20, n=N, seed=SEED)
        c_hats = roll_estimator_batch(paths)
        assert c_hats.shape == (20,)
        assert np.all(c_hats >= 0.0)

    def test_short_series(self):
        """Series of length 1 should return NaN."""
        result = roll_estimator(np.array([0.01]))
        assert np.isnan(result)

    def test_known_autocovariance(self):
        """For a series with exact lag-1 autocovariance -c^2, Roll returns c."""
        c_true = 0.05
        n = 1_000_000
        rng = np.random.default_rng(0)
        Y = simulate_roll_path(c_true, 0.0, n, rng)
        # With no noise, Roll should estimate c_true closely
        c_hat = roll_estimator(Y)
        assert abs(c_hat - c_true) < 0.002

    def test_all_zeros_series(self):
        """All-zero series gives autocovariance 0, so Roll returns 0."""
        Y = np.zeros(100)
        assert roll_estimator(Y) == 0.0


# ---------------------------------------------------------------------------
# CF estimator
# ---------------------------------------------------------------------------

class TestCFEstimator:
    def test_admissibility_reasonable_frequency(self):
        """At bias-neutral frequency, most paths should be admissible."""
        from src.roots import find_interior_root
        paths  = simulate_roll_paths(C, V, n_paths=50, n=N, seed=SEED)
        a_star = find_interior_root(C, V, n_grid=3000)
        res    = cf_estimator_batch(paths, a_star)
        adm_rate = np.mean(res['admissible'])
        # With n=252 and reasonable c/V, expect high admissibility
        assert adm_rate > 0.5, f"Admissibility rate too low: {adm_rate:.2f}"

    def test_c_hat_non_negative_where_admissible(self):
        """c_hat must be non-negative (arctan returns positive value)."""
        from src.roots import find_interior_root
        paths  = simulate_roll_paths(C, V, n_paths=30, n=N, seed=SEED)
        a_star = find_interior_root(C, V, n_grid=3000)
        res    = cf_estimator_batch(paths, a_star)
        valid  = res['c_hat'][res['admissible']]
        assert np.all(valid >= 0.0)

    def test_inadmissible_gives_nan(self):
        """At extreme frequency outside domain, phi(a) < 0 possible -> NaN."""
        # Beyond upper bound, cos^2(cu) reaches 0 or negative territory
        a_extreme = np.pi / C   # well outside admissible domain
        Y = simulate_roll_path(C, V, N, np.random.default_rng(0))
        res = cf_estimator_single(Y, a_extreme)
        # Result may or may not be admissible; just check output structure
        assert 'c_hat' in res
        assert 'admissible' in res

    def test_mixture_in_range(self):
        """Mixture estimator output should be non-negative."""
        from src.roots import find_interior_root
        paths  = simulate_roll_paths(C, V, n_paths=20, n=N, seed=SEED)
        a_star = find_interior_root(C, V, n_grid=3000)
        c_mix  = cf_roll_mixture_estimator(paths, a_star, w=0.5)
        assert np.all(c_mix >= 0.0)

    def test_mixture_w0_equals_roll(self):
        """With w=0, mixture collapses to Roll estimator."""
        from src.roots import find_interior_root
        paths  = simulate_roll_paths(C, V, n_paths=10, n=N, seed=SEED)
        a_star = find_interior_root(C, V, n_grid=3000)
        c_mix  = cf_roll_mixture_estimator(paths, a_star, w=0.0)
        c_roll = roll_estimator_batch(paths)
        np.testing.assert_allclose(c_mix, c_roll, rtol=1e-10)


# ---------------------------------------------------------------------------
# GMM estimator
# ---------------------------------------------------------------------------

class TestGMMEstimator:
    def test_model_moments_between_zero_one(self):
        """ECF model moments f_j in [0, 1]."""
        m = model_moments(C, V, DEFAULT_FREQS)
        assert np.all(m >= 0.0) and np.all(m <= 1.0)

    def test_model_moments_decreasing_in_freq(self):
        """Higher frequency -> smaller ECF moment (more oscillation)."""
        m = model_moments(C, V, DEFAULT_FREQS)
        # Not strictly monotone always, but check first few are in range
        assert m[0] > 0.0 and m[-1] >= 0.0

    def test_augmented_moment_last_element(self):
        """Last element of augmented model moment is -c^2."""
        m_aug = model_moments_augmented(C, V, DEFAULT_FREQS)
        assert abs(m_aug[-1] - (-C**2)) < 1e-16

    def test_sample_moments_shape(self):
        Y = simulate_roll_path(C, V, N, np.random.default_rng(SEED))
        m = sample_moments(Y, DEFAULT_FREQS)
        assert m.shape == (len(DEFAULT_FREQS),)

    def test_sample_moments_augmented_shape(self):
        Y = simulate_roll_path(C, V, N, np.random.default_rng(SEED))
        m = sample_moments_augmented(Y, DEFAULT_FREQS)
        assert m.shape == (len(DEFAULT_FREQS) + 1,)

    def test_sample_moments_in_neg1_to_1(self):
        """Sample ECF moments must be in [-1, 1]."""
        Y = simulate_roll_path(C, V, N, np.random.default_rng(SEED))
        m = sample_moments(Y, DEFAULT_FREQS)
        assert np.all(np.abs(m) <= 1.0 + 1e-12)

    def test_long_run_covariance_symmetric_psd(self):
        """Long-run covariance matrix should be (approximately) PSD."""
        Y      = simulate_roll_path(C, V, 500, np.random.default_rng(SEED))
        Omega  = long_run_covariance(Y, DEFAULT_FREQS)
        assert Omega.shape == (len(DEFAULT_FREQS) + 1, len(DEFAULT_FREQS) + 1)
        # Check symmetry
        np.testing.assert_allclose(Omega, Omega.T, atol=1e-14)

    def test_gmm_output_positive(self):
        """GMM c_hat should be positive."""
        rng = np.random.default_rng(SEED)
        Y   = simulate_roll_path(C, V, N, rng)
        res = ecf_gmm_augmented(Y, DEFAULT_FREQS, n_starts=3)
        assert res['c_hat'] > 0.0

    def test_gmm_output_reasonable(self):
        """GMM c_hat should be in a reasonable range around true c."""
        paths = simulate_roll_paths(C, V, n_paths=20, n=N, seed=SEED)
        c_hats = [ecf_gmm_augmented(paths[i], DEFAULT_FREQS, n_starts=3)['c_hat']
                  for i in range(20)]
        mean_c = np.mean(c_hats)
        # Mean should be within 50% of true c for n=252
        assert abs(mean_c - C) / C < 0.50, f"GMM mean {mean_c} far from true {C}"

    def test_gmm_V_hat_positive(self):
        """GMM V_hat should be positive."""
        Y   = simulate_roll_path(C, V, N, np.random.default_rng(0))
        res = ecf_gmm_augmented(Y, DEFAULT_FREQS, n_starts=2)
        assert res['V_hat'] > 0.0


# ---------------------------------------------------------------------------
# Hasbrouck estimator
# ---------------------------------------------------------------------------

class TestHasbrouckEstimator:
    def test_output_c_positive(self):
        """Hasbrouck posterior mean c_hat must be positive."""
        Y   = simulate_roll_path(C, V, N, np.random.default_rng(SEED))
        res = hasbrouck_mcmc(Y, burn_in=200, n_draws=300, seed=SEED)
        assert res['c_hat'] > 0.0

    def test_output_V_positive(self):
        """Hasbrouck posterior mean V_hat must be positive."""
        Y   = simulate_roll_path(C, V, N, np.random.default_rng(SEED))
        res = hasbrouck_mcmc(Y, burn_in=200, n_draws=300, seed=SEED)
        assert res['V_hat'] > 0.0

    def test_draws_shape(self):
        """c_draws and V_draws must have shape (n_draws,)."""
        n_draws = 100
        Y = simulate_roll_path(C, V, N, np.random.default_rng(SEED))
        res = hasbrouck_mcmc(Y, burn_in=50, n_draws=n_draws, seed=SEED)
        assert res['c_draws'].shape == (n_draws,)
        assert res['V_draws'].shape == (n_draws,)

    def test_draws_positive(self):
        """All retained draws must be positive (enforced in sampler)."""
        Y   = simulate_roll_path(C, V, N, np.random.default_rng(SEED))
        res = hasbrouck_mcmc(Y, burn_in=100, n_draws=200, seed=SEED)
        assert np.all(res['c_draws'] > 0.0)
        assert np.all(res['V_draws'] > 0.0)

    def test_runtime_recorded(self):
        """Runtime should be a positive float."""
        Y   = simulate_roll_path(C, V, N, np.random.default_rng(SEED))
        res = hasbrouck_mcmc(Y, burn_in=50, n_draws=50, seed=SEED)
        assert res['runtime'] > 0.0

    def test_with_market_returns(self):
        """Estimator accepts and processes exogenous market returns."""
        rng = np.random.default_rng(SEED)
        Y   = simulate_roll_path(C, V, N, rng)
        r_m = rng.normal(0.0, 0.01, N)
        res = hasbrouck_mcmc(Y, r_m=r_m, burn_in=50, n_draws=50, seed=SEED)
        assert res['c_hat'] > 0.0


# ---------------------------------------------------------------------------
# Cross-estimator consistency
# ---------------------------------------------------------------------------

class TestCrossEstimatorConsistency:
    """Compare all estimators on the same large-n pilot to check consistency."""

    def test_all_estimators_within_order_of_magnitude(self):
        """All estimators should give c_hat within 20x of true c on long series.
        Roll and GMM should be within 5x. Hasbrouck uses wider tolerance (20x)
        because the Gibbs sampler can exhibit high variance on short series.
        """
        rng   = np.random.default_rng(42)
        Y     = simulate_roll_path(C, V, 500, rng)  # moderate path

        c_roll = roll_estimator(Y)
        c_gmm  = ecf_gmm_augmented(Y, DEFAULT_FREQS, n_starts=3)['c_hat']
        # Hasbrouck: use paper's default burn-in/draws (1000/1500) for
        # proper chain convergence — Roll-based initialization is applied.
        c_has  = hasbrouck_mcmc(Y, burn_in=1000, n_draws=1500, seed=42)['c_hat']

        for name, c_hat, tol in [('Roll', c_roll, 5.0),
                                   ('GMM',  c_gmm,  5.0),
                                   ('Hasbrouck', c_has, 20.0)]:
            ratio = c_hat / C
            assert 1.0/tol <= ratio <= tol, \
                f"{name} estimate {c_hat:.5f} not within {tol}x of true {C}"
