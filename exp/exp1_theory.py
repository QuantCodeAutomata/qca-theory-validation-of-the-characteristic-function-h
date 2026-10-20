"""
Experiment 1: Theory validation of the CF half-spread estimator under Roll's model.

Validates analytical formulas for:
  - CF identification identity: 1 - sqrt(phi(2a))/phi(a)^2 = tan^4(ac)
  - Transformed-bias curves T(b) vs frequency a over SNR grid 0.40-1.00
  - Interior bias-neutral frequency a* (sign change of b)
  - CF variance coefficient sigma_CF^2 vs frequency
  - CF-Roll convex mixture improvement: R > G condition and optimal w*
  - Appendix validation: z = a*c at rho=1 (expected ~0.67855232)
                                      and rho=2 (expected ~0.69126736)

No external data are used. All computations are formula-driven.
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.theory import (
    phi, cf_identification_lhs, cf_identification_rhs,
    cf_bias_coeff, cf_variance_coeff, transform_bias,
    roll_asymptotic, cf_roll_mixture, root_validation_equation,
)
from src.roots import (
    build_frequency_grid, find_bias_neutral_frequency, find_interior_root
)
from src.plotting import (
    plot_transformed_bias_curves, plot_variance_tradeoff, plot_mixture_weights
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

C = 0.02
SNR_GRID = np.arange(0.400, 1.001, 0.025)   # 0.400, 0.425, ..., 1.000
N_GRID   = 5000   # dense grid for stable root detection


def run_identification_identity_check(c: float = C, n_samples: int = 200) -> dict:
    """
    Step 1: Validate identity 1 - sqrt(phi(2a))/phi(a)^2 = tan^4(ac)
    using randomly sampled admissible frequencies.
    """
    rng = np.random.default_rng(0)
    a_upper = np.pi / (4.0 * c)
    a_samples = rng.uniform(1e-4 / c, 0.999 * a_upper, n_samples)
    V = c**2   # rho=1 for the identity check (V cancels in the ratio)

    lhs = cf_identification_lhs(a_samples, c, V)
    rhs = cf_identification_rhs(a_samples, c)

    max_err = float(np.nanmax(np.abs(lhs - rhs)))
    print(f"[Exp1] CF identification identity max error: {max_err:.3e}  (should be <1e-10)")
    assert max_err < 1e-10, f"Identity check failed: max_err={max_err}"
    return {'identity_max_error': max_err, 'n_samples': n_samples}


def run_bias_curves(c: float = C) -> tuple:
    """
    Steps 2-8: Build and save transformed-bias Figure 1 curves.

    Returns (DataFrame of bias sign info, figure)
    """
    a_grid  = build_frequency_grid(c, N_GRID)
    a_upper = np.pi / (4.0 * c)
    rows = []
    for snr in SNR_GRID:
        V  = c**2 / snr
        b  = cf_bias_coeff(a_grid, c, V)
        # Signs near lower and upper boundaries
        b_finite = b[np.isfinite(b)]
        sign_low  = int(np.sign(b_finite[0]))  if len(b_finite) > 0 else 0
        sign_high = int(np.sign(b_finite[-1])) if len(b_finite) > 0 else 0
        rows.append({'snr': snr, 'V': V, 'sign_near_0': sign_low,
                     'sign_near_bound': sign_high})

    df_signs = pd.DataFrame(rows)
    print("[Exp1] Bias sign near lower/upper boundary per SNR:")
    print(df_signs[['snr', 'sign_near_0', 'sign_near_bound']].to_string(index=False))

    fig = plot_transformed_bias_curves(SNR_GRID, c, N_GRID)
    save_path = os.path.join(RESULTS_DIR, 'exp1_fig1_transformed_bias.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Exp1] Figure 1 saved to {save_path}")
    return df_signs, fig


def run_root_finding(c: float = C) -> pd.DataFrame:
    """
    Steps 9-10: Detect bias-neutral frequencies a* for each SNR.
    Validates roots against the equivalent algebraic equation.
    """
    rows = []
    for snr in SNR_GRID:
        V = c**2 / snr
        result = find_bias_neutral_frequency(c, V, N_GRID)
        roots = result['roots']

        if roots:
            a_star = roots[-1]   # interior bias-neutral root (largest)
            z_star = a_star * c
            # Validate: root_validation_equation should be ~0
            eq_val = float(root_validation_equation(np.array([a_star]), c, V)[0])
            # Verify root is interior
            in_domain = (0.0 < a_star < np.pi / (4.0 * c))
        else:
            a_star = np.nan
            z_star = np.nan
            eq_val = np.nan
            in_domain = False

        rows.append({
            'snr': round(float(snr), 3),
            'V':   round(float(c**2 / snr), 8),
            'n_roots':   len(roots),
            'a_star':    round(float(a_star), 8) if not np.isnan(a_star) else np.nan,
            'z_star':    round(float(z_star), 8) if not np.isnan(z_star) else np.nan,
            'eq_residual': round(float(eq_val), 10) if np.isfinite(eq_val) else np.nan,
            'in_domain': in_domain,
        })

    df = pd.DataFrame(rows)
    print("\n[Exp1] Bias-neutral frequencies a* per SNR:")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, 'exp1_roots.csv'), index=False)
    return df


def run_variance_analysis(c: float = C, df_roots: pd.DataFrame = None) -> pd.DataFrame:
    """
    Steps 6, 11-13: Evaluate sigma_CF^2, Roll asymptotic, and CF-Roll mixture w*.
    """
    rows = []
    for snr in SNR_GRID:
        V = c**2 / snr
        a_star = find_interior_root(c, V, N_GRID)

        if np.isnan(a_star):
            rows.append({'snr': round(snr, 3), 'a_star': np.nan,
                         'sigma2_CF': np.nan, 'R': np.nan, 'G': np.nan,
                         'C': np.nan, 'D_star': np.nan, 'w_star': np.nan,
                         'improvement': False})
            continue

        sigma2_CF = float(cf_variance_coeff(np.array([a_star]), c, V)[0])
        mix = cf_roll_mixture(a_star, c, V)

        rows.append({
            'snr':      round(float(snr), 3),
            'a_star':   round(float(a_star), 8),
            'sigma2_CF': round(float(sigma2_CF), 8),
            'R':        round(float(mix['R']), 8),
            'G':        round(float(mix['G']), 8),
            'C':        round(float(mix['C']), 8),
            'D_star':   round(float(mix['D_star']), 8),
            'w_star':   round(float(mix['w_star']), 8),
            'improvement': mix['improvement'],
        })

    df = pd.DataFrame(rows)
    print("\n[Exp1] CF variance and mixture weight summary:")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, 'exp1_mixture_summary.csv'), index=False)

    # Plot w* vs SNR
    fig_w = plot_mixture_weights(df)
    fig_w.savefig(os.path.join(RESULTS_DIR, 'exp1_mixture_weight.png'), dpi=150)
    plt.close(fig_w)
    return df


def run_appendix_validation(c: float = C) -> pd.DataFrame:
    """
    Step 14: Appendix validation at rho=1 (V=c^2) and rho=2 (V=2c^2).

    Expected z=a*c values: rho=1 -> 0.67855232, rho=2 -> 0.69126736
    """
    appendix_cases = [
        {'rho': 1.0, 'expected_z': 0.67855232},
        {'rho': 2.0, 'expected_z': 0.69126736},
    ]
    rows = []
    for case in appendix_cases:
        rho = case['rho']
        V   = rho * c**2
        a_star = find_interior_root(c, V, n_grid=10000)
        z_star = a_star * c if not np.isnan(a_star) else np.nan
        err    = abs(z_star - case['expected_z']) if np.isfinite(z_star) else np.nan

        sigma2 = float(cf_variance_coeff(np.array([a_star]), c, V)[0]) if not np.isnan(a_star) else np.nan
        mix = cf_roll_mixture(a_star, c, V) if not np.isnan(a_star) else {}

        rows.append({
            'rho':        rho,
            'expected_z': case['expected_z'],
            'computed_z': round(float(z_star), 8) if np.isfinite(z_star) else np.nan,
            'abs_error':  round(float(err), 10) if np.isfinite(err) else np.nan,
            'R':          round(float(mix.get('R', np.nan)), 8),
            'G':          round(float(mix.get('G', np.nan)), 8),
            'C':          round(float(mix.get('C', np.nan)), 8),
            'w_star':     round(float(mix.get('w_star', np.nan)), 8),
        })
        print(f"[Exp1] Appendix rho={rho}: expected z={case['expected_z']}, "
              f"computed z={z_star:.8f}, error={err:.2e}")

    df_app = pd.DataFrame(rows)
    df_app.to_csv(os.path.join(RESULTS_DIR, 'exp1_appendix_validation.csv'), index=False)
    return df_app


def run_variance_tradeoff_plot(c: float = C) -> None:
    """Plot sigma_CF^2 for representative (c, V) pairs."""
    rho_list = [0.5, 1.0, 2.0]
    V_list   = [rho * c**2 for rho in rho_list]
    labels   = [f'rho={rho}' for rho in rho_list]

    fig = plot_variance_tradeoff(c, V_list, labels, n_grid=N_GRID)
    fig.savefig(os.path.join(RESULTS_DIR, 'exp1_variance_tradeoff.png'), dpi=150)
    plt.close(fig)
    print(f"[Exp1] Variance tradeoff plot saved.")


def main() -> None:
    print("=" * 60)
    print("Experiment 1: Theory Validation (Roll CF Estimator)")
    print("=" * 60)

    # Step 1: Identity check
    id_result = run_identification_identity_check()

    # Steps 2-8: Transformed-bias curves
    df_signs, _ = run_bias_curves()

    # Steps 9-10: Root finding
    df_roots = run_root_finding()

    # Steps 6, 11-13: Variance and mixture analysis
    df_mixture = run_variance_analysis()

    # Step 14: Appendix validation
    df_app = run_appendix_validation()

    # Variance tradeoff plot
    run_variance_tradeoff_plot()

    print("\n[Exp1] All outputs saved to results/")
    print("[Exp1] SUMMARY:")
    print(f"  CF identity max error : {id_result['identity_max_error']:.2e}")
    print(f"  Roots found for all SNR: {df_roots['n_roots'].gt(0).all()}")
    print(f"  All roots in domain   : {df_roots['in_domain'].all()}")
    app_ok = all(row['abs_error'] < 1e-5 for _, row in df_app.iterrows() if np.isfinite(row['abs_error']))
    print(f"  Appendix validation   : {'PASS' if app_ok else 'CHECK TOLERANCE'}")
    for _, row in df_app.iterrows():
        print(f"    rho={row['rho']}: z_computed={row['computed_z']}, "
              f"expected={row['expected_z']}, error={row['abs_error']:.2e}")


if __name__ == '__main__':
    main()
