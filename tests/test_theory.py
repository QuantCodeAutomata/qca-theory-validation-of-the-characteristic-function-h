"""
Tests for src/theory.py and src/roots.py.

Validates:
  - CF identification identity
  - Sign pattern of bias coefficient b(a; c, V)
  - Interior bias-neutral root a* existence and domain constraint
  - Appendix benchmark values (rho=1 z~0.67855232, rho=2 z~0.69126736)
  - CF variance coefficient positivity
  - Roll asymptotic formulas
  - CF-Roll mixture computation
"""

import numpy as np
import pytest

from src.theory import (
    phi, cf_identification_lhs, cf_identification_rhs,
    cf_bias_coeff, cf_variance_coeff, transform_bias,
    roll_asymptotic, cf_roll_mixture, root_validation_equation,
    covariance_terms, compute_B_K, xi_function,
)
from src.roots import (
    build_frequency_grid, detect_sign_changes,
    find_bias_neutral_frequency, find_interior_root,
)

C = 0.02


# ---------------------------------------------------------------------------
# CF characteristic function
# ---------------------------------------------------------------------------

class TestPhi:
    def test_phi_at_zero_is_one(self):
        """phi(0) = 1 for any c, V."""
        result = phi(np.array([0.0]), C, 0.0001)
        assert abs(result[0] - 1.0) < 1e-12

    def test_phi_positive_in_admissible_domain(self):
        """phi(a) > 0 for 0 < a*c < pi/4."""
        a_grid = build_frequency_grid(C, 200)
        vals   = phi(a_grid, C, C**2)
        assert np.all(vals > 0), "phi(a) must be positive in admissible domain"

    def test_phi_monotone_decreasing_in_V(self):
        """Larger V gives smaller phi (more noise dampens CF)."""
        a = np.array([5.0])
        p_low  = phi(a, C, 0.0001)
        p_high = phi(a, C, 0.01)
        assert p_low > p_high

    def test_phi_between_zero_and_one(self):
        """phi must be in [0, 1] everywhere."""
        u = np.linspace(0, 80, 500)
        vals = phi(u, C, C**2)
        assert np.all(vals >= 0.0)
        assert np.all(vals <= 1.0 + 1e-12)


# ---------------------------------------------------------------------------
# CF identification identity: 1 - sqrt(phi(2a))/phi(a)^2 = tan^4(ac)
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_identity_holds_at_random_admissible_a(self):
        """LHS equals RHS to machine precision for 500 sampled admissible a."""
        rng = np.random.default_rng(42)
        a_upper = np.pi / (4.0 * C)
        a_samples = rng.uniform(1e-3 / C, 0.99 * a_upper, 500)
        V = C**2   # rho=1

        lhs = cf_identification_lhs(a_samples, C, V)
        rhs = cf_identification_rhs(a_samples, C)

        max_err = float(np.nanmax(np.abs(lhs - rhs)))
        assert max_err < 1e-10, f"Identity max error {max_err} exceeds tolerance"

    def test_identity_V_cancels(self):
        """Identity should hold for any V > 0 (V cancels in ratio)."""
        a = np.array([10.0])
        for V in [1e-6, C**2, 5 * C**2, 0.1]:
            lhs = cf_identification_lhs(a, C, V)
            rhs = cf_identification_rhs(a, C)
            assert abs(lhs[0] - rhs[0]) < 1e-10, f"Identity failed for V={V}"

    def test_rhs_in_zero_one(self):
        """tan^4(ac) should be in (0,1) for 0 < ac < pi/4."""
        a_grid = build_frequency_grid(C, 100)
        rhs = cf_identification_rhs(a_grid, C)
        assert np.all(rhs > 0)
        assert np.all(rhs < 1.0)


# ---------------------------------------------------------------------------
# Bias coefficient sign pattern
# ---------------------------------------------------------------------------

class TestBiasCoeff:
    @pytest.mark.parametrize("snr", [0.40, 0.60, 0.80, 1.00])
    def test_bias_negative_near_lower_boundary(self, snr: float):
        """b(a;c,V) < 0 near a -> 0+."""
        V      = C**2 / snr
        a_grid = build_frequency_grid(C, 5000)
        b      = cf_bias_coeff(a_grid, C, V)
        b_finite = b[np.isfinite(b)]
        assert len(b_finite) > 0, "No finite bias values"
        assert b_finite[0] < 0.0, (
            f"Bias should be negative near lower boundary for SNR={snr}, "
            f"got {b_finite[0]}"
        )

    @pytest.mark.parametrize("snr", [0.40, 0.60, 0.80, 1.00])
    def test_bias_positive_near_upper_boundary(self, snr: float):
        """b(a;c,V) > 0 near a -> pi/(4c)-."""
        V      = C**2 / snr
        a_grid = build_frequency_grid(C, 5000)
        b      = cf_bias_coeff(a_grid, C, V)
        b_finite = b[np.isfinite(b)]
        assert len(b_finite) > 0, "No finite bias values"
        assert b_finite[-1] > 0.0, (
            f"Bias should be positive near upper boundary for SNR={snr}, "
            f"got {b_finite[-1]}"
        )

    def test_transform_bias_preserves_sign(self):
        """T(b) = sign(b)*log10(1+|b|/c) preserves sign of b."""
        b = np.array([-0.05, -0.001, 0.0, 0.001, 0.05])
        T = transform_bias(b, b0=C)
        assert np.sign(T[0]) == -1
        assert np.sign(T[-1]) == 1

    def test_transform_bias_at_zero(self):
        """T(0) = 0."""
        assert abs(transform_bias(np.array([0.0]))[0]) < 1e-15

    def test_transform_bias_monotone(self):
        """T(b) is monotonically increasing in b for b >= 0."""
        b = np.linspace(0.0, 0.1, 50)
        T = transform_bias(b, b0=C)
        diffs = np.diff(T)
        assert np.all(diffs >= 0), "T(b) must be non-decreasing"


# ---------------------------------------------------------------------------
# Root finding: bias-neutral frequency a*
# ---------------------------------------------------------------------------

class TestRoots:
    @pytest.mark.parametrize("snr", [0.40, 0.60, 0.80, 1.00])
    def test_interior_root_exists(self, snr: float):
        """An interior bias-neutral root a* must exist for all tested SNR."""
        V  = C**2 / snr
        a_star = find_interior_root(C, V, n_grid=5000)
        assert np.isfinite(a_star), f"No root found for SNR={snr}"

    @pytest.mark.parametrize("snr", [0.40, 0.60, 0.80, 1.00])
    def test_root_in_admissible_domain(self, snr: float):
        """a* must lie strictly inside (0, pi/(4c))."""
        V      = C**2 / snr
        a_star = find_interior_root(C, V, n_grid=5000)
        a_upper = np.pi / (4.0 * C)
        assert 0.0 < a_star < a_upper, (
            f"Root {a_star} outside (0, {a_upper}) for SNR={snr}"
        )

    @pytest.mark.parametrize("snr", [0.40, 0.60, 0.80, 1.00])
    def test_bias_near_zero_at_root(self, snr: float):
        """b(a*;c,V) should be near zero at the detected root."""
        V      = C**2 / snr
        a_star = find_interior_root(C, V, n_grid=5000)
        b_root = float(cf_bias_coeff(np.array([a_star]), C, V)[0])
        assert abs(b_root) < 1e-8, (
            f"|b(a*)| = {abs(b_root):.2e} not near zero for SNR={snr}"
        )

    def test_root_validation_equation_at_root(self):
        """Root validation equation should evaluate to ~0 at a*."""
        V      = C**2 / 0.70
        a_star = find_interior_root(C, V, n_grid=5000)
        val    = float(root_validation_equation(np.array([a_star]), C, V)[0])
        assert abs(val) < 1e-5, f"Root validation equation residual {val:.2e} too large"

    def test_appendix_rho1(self):
        """Appendix: rho=1 gives z=a*c ~ 0.67855232."""
        rho  = 1.0
        V    = rho * C**2
        a_star = find_interior_root(C, V, n_grid=10000)
        z_star = a_star * C
        expected = 0.67855232
        assert abs(z_star - expected) < 1e-4, (
            f"rho=1: z={z_star:.8f}, expected {expected}, error={abs(z_star-expected):.2e}"
        )

    def test_appendix_rho2(self):
        """Appendix: rho=2 gives z=a*c ~ 0.69126736."""
        rho  = 2.0
        V    = rho * C**2
        a_star = find_interior_root(C, V, n_grid=10000)
        z_star = a_star * C
        expected = 0.69126736
        assert abs(z_star - expected) < 1e-4, (
            f"rho=2: z={z_star:.8f}, expected {expected}, error={abs(z_star-expected):.2e}"
        )


# ---------------------------------------------------------------------------
# CF variance coefficient
# ---------------------------------------------------------------------------

class TestVarianceCoeff:
    def test_variance_positive(self):
        """sigma_CF^2 > 0 in admissible domain."""
        V      = C**2
        a_grid = build_frequency_grid(C, 100)
        sigma2 = cf_variance_coeff(a_grid, C, V)
        finite = sigma2[np.isfinite(sigma2)]
        assert np.all(finite > 0), "Variance coefficient must be positive"

    def test_variance_increases_near_boundary(self):
        """Variance should increase as a approaches upper boundary."""
        V = C**2
        a_grid = build_frequency_grid(C, 1000)
        sigma2 = cf_variance_coeff(a_grid, C, V)
        finite = sigma2[np.isfinite(sigma2)]
        # Last 10% should have larger variance than first 10% on average
        n = len(finite)
        assert np.mean(finite[int(0.9*n):]) > np.mean(finite[:int(0.1*n)]), \
            "Variance should increase toward upper boundary"


# ---------------------------------------------------------------------------
# Roll asymptotic formulas
# ---------------------------------------------------------------------------

class TestRollAsymptotic:
    def test_L_R_formula(self):
        """L_R = V^2 + 4Vc^2 + 5c^4."""
        V   = C**2
        res = roll_asymptotic(C, V)
        expected_L = V**2 + 4*V*C**2 + 5*C**4
        assert abs(res['L_R'] - expected_L) < 1e-18

    def test_R_positive(self):
        """R = L_R / (4c^4) > 0."""
        V = C**2
        res = roll_asymptotic(C, V)
        assert res['R'] > 0.0

    def test_B_R_negative(self):
        """B_R = -L_R/(8c^3) < 0 (Roll is negatively biased)."""
        V = C**2
        res = roll_asymptotic(C, V)
        assert res['B_R'] < 0.0


# ---------------------------------------------------------------------------
# CF-Roll mixture
# ---------------------------------------------------------------------------

class TestMixture:
    def test_w_star_in_unit_interval(self):
        """Optimal weight w* must be in [0, 1]."""
        V      = C**2
        a_star = find_interior_root(C, V, n_grid=3000)
        res    = cf_roll_mixture(a_star, C, V)
        assert 0.0 <= res['w_star'] <= 1.0, \
            f"w* = {res['w_star']} outside [0,1]"

    def test_C_positive(self):
        """CF variance coefficient C should be positive."""
        V      = C**2
        a_star = find_interior_root(C, V, n_grid=3000)
        res    = cf_roll_mixture(a_star, C, V)
        assert res['C'] > 0.0

    def test_R_positive(self):
        """Roll asymptotic R should be positive."""
        V      = C**2
        a_star = find_interior_root(C, V, n_grid=3000)
        res    = cf_roll_mixture(a_star, C, V)
        assert res['R'] > 0.0


# ---------------------------------------------------------------------------
# Covariance terms and numerical guards
# ---------------------------------------------------------------------------

class TestCovarianceTerms:
    def test_S11_non_negative(self):
        """Variance S11 = Var(cos(aY)) >= 0."""
        a_grid = build_frequency_grid(C, 200)
        S11, _, _ = covariance_terms(a_grid, C, C**2)
        assert np.all(S11[np.isfinite(S11)] >= -1e-12)

    def test_S22_non_negative(self):
        """Variance S22 = Var(cos(2aY)) >= 0."""
        a_grid = build_frequency_grid(C, 200)
        _, _, S22 = covariance_terms(a_grid, C, C**2)
        assert np.all(S22[np.isfinite(S22)] >= -1e-12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_small_c(self):
        """Small c should still yield valid root (domain scales with c)."""
        c_small = 0.001
        V       = c_small**2
        a_star  = find_interior_root(c_small, V, n_grid=3000)
        assert np.isfinite(a_star)
        assert 0.0 < a_star * c_small < np.pi / 4.0

    def test_large_snr(self):
        """SNR=2 (low noise) should still yield admissible root."""
        V      = C**2 / 2.0
        a_star = find_interior_root(C, V, n_grid=5000)
        assert np.isfinite(a_star)

    def test_small_snr(self):
        """SNR=0.40 (high noise) should still yield admissible root."""
        V      = C**2 / 0.40
        a_star = find_interior_root(C, V, n_grid=5000)
        assert np.isfinite(a_star)

    def test_xi_function_positive(self):
        """Xi(u) >= 0 for all u in admissible domain."""
        a_grid = build_frequency_grid(C, 200)
        xi = xi_function(a_grid, C, C**2)
        assert np.all(xi[np.isfinite(xi)] >= -1e-12)


# ---------------------------------------------------------------------------
# Regression tests with known expected values
# ---------------------------------------------------------------------------

class TestRegressionValues:
    def test_phi_at_specific_values(self):
        """phi(u=1/(4c), c=0.02, V=c^2) regression."""
        # At u = pi/(4c), cos^2(c*u) = cos^2(pi/4) = 0.5
        u_val = np.pi / (4.0 * C)
        result = phi(np.array([u_val]), C, C**2)
        # Expected: exp(-c^2 * (pi/(4c))^2 / 2) * cos^2(pi/4)
        gauss  = np.exp(-C**2 * (np.pi / (4*C))**2 / 2.0)
        expected = gauss * 0.5
        assert abs(result[0] - expected) < 1e-14

    def test_roll_asymptotic_at_rho1(self):
        """Roll asymptotic R at rho=1 regression value."""
        rho = 1.0
        V   = rho * C**2
        res = roll_asymptotic(C, V)
        # L_R = (C^2)^2 + 4*C^2*C^2 + 5*C^4 = 10*C^4
        expected_L = 10.0 * C**4
        assert abs(res['L_R'] - expected_L) < 1e-20
        expected_R = expected_L / (4.0 * C**4)
        assert abs(res['R'] - expected_R) < 1e-14

    def test_bias_neutral_z_snr1(self):
        """Appendix: z=a*c at SNR=1 (rho=1) is ~0.67855232."""
        V = C**2
        a_star = find_interior_root(C, V, n_grid=10000)
        z = a_star * C
        assert abs(z - 0.67855232) < 5e-5, f"z={z:.8f}"
