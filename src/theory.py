"""
Core characteristic-function (CF) half-spread estimator theory under Roll's model.

Roll model:
  Y_t = c*(Q_t - Q_{t-1}) + U_t
  Q_t in {-1, +1} iid symmetric, P(Q_t=1)=0.5
  U_t ~ N(0, V) iid, independent of Q
  phi(u) = E[exp(i*u*Y_t)] real part = exp(-V*u^2/2) * cos^2(c*u)

All computations use float64.
Custom implementation -- Context7 confirmed no library equivalent for these formulas.
"""

import numpy as np


def phi(u: np.ndarray, c: float, V: float) -> np.ndarray:
    """
    Roll model characteristic function (real part) of Y_t.

    phi(u) = exp(-V*u^2/2) * cos^2(c*u)

    Parameters
    ----------
    u : array-like, frequency argument
    c : float, half-spread parameter (c > 0)
    V : float, noise variance (V > 0)

    Returns
    -------
    np.ndarray, values in [0, 1] for admissible u
    """
    u = np.asarray(u, dtype=np.float64)
    return np.exp(-V * u**2 / 2.0) * np.cos(c * u)**2


def cf_identification_lhs(a: np.ndarray, c: float, V: float) -> np.ndarray:
    """
    Compute LHS of CF identification identity:
      LHS = 1 - sqrt(phi(2a)) / phi(a)^2

    This equals tan^4(ac) by the CF identification theorem (V cancels).

    Parameters
    ----------
    a : array-like, frequency grid (admissible: 0 < a*c < pi/4)
    c : float, half-spread
    V : float, noise variance
    """
    a = np.asarray(a, dtype=np.float64)
    x = phi(a, c, V)     # phi(a)
    y = phi(2*a, c, V)   # phi(2a)
    # Guard: x^2 > 0 required
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.sqrt(np.maximum(y, 0.0)) / np.where(x > 0, x**2, np.nan)
    return 1.0 - r


def cf_identification_rhs(a: np.ndarray, c: float) -> np.ndarray:
    """
    Compute RHS of CF identification identity: tan^4(a*c).

    Parameters
    ----------
    a : array-like, frequency grid
    c : float, half-spread
    """
    a = np.asarray(a, dtype=np.float64)
    return np.tan(c * a)**4


def covariance_terms(a: np.ndarray, c: float, V: float) -> tuple:
    """
    Compute variance-covariance terms for (cos(aY), cos(2aY)) under Roll's model.

    S11 = Var(cos(aY))  = (1 + phi(2a))/2 - phi(a)^2
    S12 = Cov(cos(aY), cos(2aY)) = (phi(a) + phi(3a))/2 - phi(a)*phi(2a)
    S22 = Var(cos(2aY)) = (1 + phi(4a))/2 - phi(2a)^2

    These follow from the product-to-sum identity for the marginal distribution.

    Parameters
    ----------
    a : array-like
    c, V : Roll model parameters

    Returns
    -------
    tuple (S11, S12, S22), each np.ndarray
    """
    a = np.asarray(a, dtype=np.float64)
    p1 = phi(a, c, V)
    p2 = phi(2*a, c, V)
    p3 = phi(3*a, c, V)
    p4 = phi(4*a, c, V)

    S11 = (1.0 + p2) / 2.0 - p1**2
    S12 = (p1 + p3) / 2.0 - p1 * p2
    S22 = (1.0 + p4) / 2.0 - p2**2
    return S11, S12, S22


def compute_B_K(a: np.ndarray, c: float, V: float) -> tuple:
    """
    Compute B and K, intermediate quantities for CF bias and variance coefficients.

    Let x = phi(a), y = phi(2a), r = sqrt(y)/x^2, d = 1 - r.

    B = r * (-3*S11/x^2 + S12/(x*y) + S22/(8*y^2))
    K = r^2 * (4*S11/x^2 - 2*S12/(x*y) + S22/(4*y^2))

    Parameters
    ----------
    a : array-like, frequency grid
    c, V : Roll model parameters

    Returns
    -------
    tuple (B, K, x, y, r, d)
    """
    a = np.asarray(a, dtype=np.float64)
    x = phi(a, c, V)
    y = phi(2*a, c, V)
    S11, S12, S22 = covariance_terms(a, c, V)

    with np.errstate(divide='ignore', invalid='ignore'):
        sqrt_y = np.sqrt(np.maximum(y, 0.0))
        x2 = np.where(x > 0, x**2, np.nan)
        r = sqrt_y / x2
        d = 1.0 - r

        # Compute ratios carefully
        S11_x2 = np.where(x > 0, S11 / x2, np.nan)
        S12_xy = np.where((x > 0) & (y > 0), S12 / (x * y), np.nan)
        S22_y2 = np.where(y > 0, S22 / y**2, np.nan)

        B = r * (-3.0*S11_x2 + S12_xy + S22_y2 / 8.0)
        K = r**2 * (4.0*S11_x2 - 2.0*S12_xy + S22_y2 / 4.0)

    return B, K, x, y, r, d


def cf_variance_coeff(a: np.ndarray, c: float, V: float) -> np.ndarray:
    """
    CF variance coefficient (asymptotic):
      sigma_CF^2(a;c,V) = K / [16 * a^2 * d^(3/2) * (1 + sqrt(d))^2]

    Parameters
    ----------
    a : array-like, frequency grid (admissible)
    c, V : Roll model parameters

    Returns
    -------
    np.ndarray, variance coefficients
    """
    a = np.asarray(a, dtype=np.float64)
    B, K, x, y, r, d = compute_B_K(a, c, V)

    # Guard: d < 1e-8 is floating-point noise (d = tan^4(ac) ≈ (ac)^4,
    # so this corresponds to ac < ~0.01), where the formula is unreliable.
    valid = (d >= 1e-8) & np.isfinite(d) & np.isfinite(K)

    with np.errstate(divide='ignore', invalid='ignore'):
        sqrt_d = np.sqrt(np.where(valid, d, 1.0))
        denom = 16.0 * a**2 * d**(1.5) * (1.0 + sqrt_d)**2
        sigma2 = np.where(valid & (denom > 0), K / denom, np.nan)
    return sigma2


def cf_bias_coeff(a: np.ndarray, c: float, V: float) -> np.ndarray:
    """
    CF bias coefficient (leading finite-sample term):
      b(a;c,V) = B / [4*a*d^(3/4)*(1+sqrt(d))]
               - [(3 + 5*sqrt(d))*K] / [32*a*d^(7/4)*(1+sqrt(d))^2]

    Parameters
    ----------
    a : array-like, frequency grid (admissible: 0 < ac < pi/4)
    c, V : Roll model parameters

    Returns
    -------
    np.ndarray, bias coefficients (sign changes mark interior root a*)
    """
    a = np.asarray(a, dtype=np.float64)
    B, K, x, y, r, d = compute_B_K(a, c, V)

    # Guard: d < 1e-8 is floating-point noise near a→0 (d = tan^4(ac) → 0).
    # The formula diverges as d→0 due to d^(7/4) in the denominator; discard
    # these points to avoid spurious sign-flip at the first grid points.
    valid = (d >= 1e-8) & np.isfinite(d) & np.isfinite(B) & np.isfinite(K)

    with np.errstate(divide='ignore', invalid='ignore'):
        sqrt_d = np.sqrt(np.where(valid, d, 1.0))
        d_safe = np.where(valid, d, 1.0)

        # Term 1: B / [4*a*d^(3/4)*(1+sqrt(d))]
        denom1 = 4.0 * a * d_safe**(0.75) * (1.0 + sqrt_d)
        term1 = np.where(valid & (denom1 > 0), B / denom1, np.nan)

        # Term 2: (3+5*sqrt(d))*K / [32*a*d^(7/4)*(1+sqrt(d))^2]
        denom2 = 32.0 * a * d_safe**(1.75) * (1.0 + sqrt_d)**2
        term2 = np.where(valid & (denom2 > 0), (3.0 + 5.0*sqrt_d) * K / denom2, np.nan)

        b = term1 - term2
    return b


def transform_bias(b: np.ndarray, b0: float = 0.02) -> np.ndarray:
    """
    Symmetrically compress bias for plotting (Figure 1 style):
      T(b) = sign(b) * log10(1 + |b| / b0)

    Parameters
    ----------
    b  : array-like, bias values
    b0 : float, scaling constant (default 0.02, matching c)
    """
    b = np.asarray(b, dtype=np.float64)
    return np.sign(b) * np.log10(1.0 + np.abs(b) / b0)


def xi_function(u: np.ndarray, c: float, V: float) -> np.ndarray:
    """
    Derivative of phi w.r.t. c (up to sign), used in CF-Roll covariance:
      Xi(u) = exp(-V*u^2/2) * {2*c^2*sin^2(c*u) + c*V*u*sin(2*c*u)}

    Parameters
    ----------
    u : array-like
    c, V : Roll model parameters
    """
    u = np.asarray(u, dtype=np.float64)
    gauss = np.exp(-V * u**2 / 2.0)
    return gauss * (2.0*c**2 * np.sin(c*u)**2 + c*V*u * np.sin(2.0*c*u))


def roll_asymptotic(c: float, V: float) -> dict:
    """
    Roll estimator asymptotic quantities:
      L_R = V^2 + 4*V*c^2 + 5*c^4
      B_R = -L_R / (8*c^3)    [asymptotic bias coefficient]
      R   = L_R / (4*c^4)     [asymptotic MSE coefficient]

    Parameters
    ----------
    c : float, half-spread
    V : float, noise variance

    Returns
    -------
    dict with keys L_R, B_R, R
    """
    L_R = V**2 + 4.0*V*c**2 + 5.0*c**4
    B_R = -L_R / (8.0 * c**3)
    R   = L_R / (4.0 * c**4)
    return {'L_R': L_R, 'B_R': B_R, 'R': R}


def root_validation_equation(a: np.ndarray, c: float, V: float) -> np.ndarray:
    """
    Independent validation of b(a;c,V)=0 via equivalent algebraic condition.

    Let p_k = phi(k*a), t = tan^2(a*c).
    Define:
      A      = (1+p2)/(2*p1^2) - 1
      D_coef = (p1+p3)/(2*p1*p2) - 1
      E_coef = (1+p4)/(2*p2^2) - 1

    Then b(a;c,V)=0 iff:
      (9t^2 - 2t - 3)*E_coef = 16*(t^2+2t+3)*A - 8*(3-t)*(1+t)*D_coef

    Returns LHS - RHS (should be ~0 at the root a*).

    Parameters
    ----------
    a : array-like
    c, V : Roll model parameters
    """
    a = np.asarray(a, dtype=np.float64)
    p1 = phi(a,   c, V)
    p2 = phi(2*a, c, V)
    p3 = phi(3*a, c, V)
    p4 = phi(4*a, c, V)
    t  = np.tan(c * a)**2

    with np.errstate(divide='ignore', invalid='ignore'):
        A      = np.where(p1 > 0, (1.0 + p2) / (2.0 * p1**2) - 1.0, np.nan)
        D_coef = np.where((p1 > 0) & (p2 > 0),
                          (p1 + p3) / (2.0 * p1 * p2) - 1.0, np.nan)
        E_coef = np.where(p2 > 0, (1.0 + p4) / (2.0 * p2**2) - 1.0, np.nan)

    LHS = (9.0*t**2 - 2.0*t - 3.0) * E_coef
    RHS = 16.0*(t**2 + 2.0*t + 3.0)*A - 8.0*(3.0 - t)*(1.0 + t)*D_coef
    return LHS - RHS


def cf_roll_mixture(a_star: float, c: float, V: float) -> dict:
    """
    Compute CF-Roll convex mixture optimal weight w* and related quantities.

    Operationalization for single-frequency case:
      Xi(u) = exp(-V*u^2/2)*{2c^2*sin^2(cu) + cVu*sin(2cu)}
      q_j = [1/(4*a*d^(3/4)*(1+sqrt(d)))] * [2r/phi(a), -r/(2*phi(2a))]
      Sigma_0j = -(q_{j,1}*Xi(a) + q_{j,2}*Xi(2a)) / (2*c)
      G = Sigma_0j / c^2

    Then:
      C     = sigma_CF^2(a*;c,V) / c^2
      R     = roll_asymptotic L_R / (4c^4)
      D_*   = C + R - 2*G
      w_*   = clip((R-G)/D_*, 0, 1)

    Parameters
    ----------
    a_star : float, bias-neutral frequency
    c      : float, half-spread
    V      : float, noise variance

    Returns
    -------
    dict with keys G, C, R, D_star, w_star, improvement (bool R > G)
    """
    a = float(a_star)

    # CF ingredients at a_star
    x  = float(phi(np.array([a]), c, V)[0])
    y  = float(phi(np.array([2*a]), c, V)[0])
    r  = np.sqrt(max(y, 0.0)) / (x**2) if x > 0 else np.nan
    d  = 1.0 - r
    sqrt_d = np.sqrt(max(d, 0.0))

    # q vector (gradient of estimator w.r.t. phi(a), phi(2a))
    coeff = 1.0 / (4.0 * a * d**0.75 * (1.0 + sqrt_d)) if d > 0 else np.nan
    q1 = coeff * 2.0 * r / x if x > 0 else np.nan
    q2 = coeff * (-r / (2.0 * y)) if y > 0 else np.nan

    # Xi at a and 2a
    xi_a  = float(xi_function(np.array([a]),   c, V)[0])
    xi_2a = float(xi_function(np.array([2*a]), c, V)[0])

    # Sigma_0j: covariance between CF estimator and Roll estimator (scaled)
    Sigma_0j = -(q1 * xi_a + q2 * xi_2a) / (2.0 * c)

    G = Sigma_0j / c**2

    # CF variance at a*
    sigma2 = float(cf_variance_coeff(np.array([a]), c, V)[0])
    C = sigma2 / c**2

    # Roll asymptotic R
    ra = roll_asymptotic(c, V)
    R  = ra['R']

    D_star = C + R - 2.0 * G
    if abs(D_star) < 1e-15:
        w_star = 0.0
    else:
        w_star = float(np.clip((R - G) / D_star, 0.0, 1.0))

    return {
        'G': G, 'C': C, 'R': R,
        'D_star': D_star, 'w_star': w_star,
        'improvement': bool(R > G),
        'Sigma_0j': Sigma_0j,
    }
