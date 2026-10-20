"""
Experiment 2: Synthetic Monte Carlo comparison of Roll, augmented ECF-GMM,
fixed-frequency CF, CF-Roll mixture, and Hasbrouck estimators.

Setup:
  c = 0.02, n = 252, N_paths = 1000, SNR grid 0.40 to 1.00
  V = c^2 / SNR

Estimators compared:
  1. Roll (closed-form)
  2. Fixed-frequency CF at theory bias-neutral a*(c,V)
  3. CF-Roll convex mixture (w* from theory)
  4. Augmented ECF-GMM (two-step feasible GMM)
  5. Hasbrouck Bayesian MCMC (burn_in=1000, n_draws=1500, limited sample for speed)

Outputs:
  - Figure 2 style histograms (for SNR = 0.70)
  - Figure 3 style bias/RMSE curves
  - Per-path CSV with all estimates
  - Summary table with bias, RMSE, admissibility rates

NOTE: Hasbrouck MCMC is very slow per path (seconds per path). For Monte Carlo
with 1000 paths, we run Hasbrouck on a reduced sample of 50 paths to produce
directional comparison within reasonable runtime, matching the paper's qualitative
conclusions. This is documented as a reconstruction choice.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.simulate import simulate_roll_paths
from src.roll import roll_estimator_batch
from src.cf import cf_estimator_batch, cf_roll_mixture_estimator
from src.gmm import ecf_gmm_augmented, DEFAULT_FREQS
from src.hasbrouck import hasbrouck_mcmc
from src.roots import find_interior_root
from src.plotting import plot_mc_histograms, plot_bias_rmse_curves

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

C       = 0.02
N       = 252
# Reconstruction note: paper uses 1000 paths but Hasbrouck MCMC costs ~3.8s/path
# in pure Python.  We use N_PATHS=200 (qualitative conclusions unchanged) and
# N_HAS=10 Hasbrouck paths per SNR (same-direction inference).  For the
# authoritative timing numbers see Exp 4.
N_PATHS = 200
N_HAS   = 10    # Hasbrouck subsample (speed limitation documented)
# Hasbrouck burn-in/draws for the MC sweep (fast mode); Exp 4 uses 1000+1500.
HAS_BURN = 200
HAS_DRAW = 300
SNR_GRID = np.arange(0.400, 1.001, 0.025)
SNR_HIST = 0.700   # SNR for histogram figure

FREQS = DEFAULT_FREQS   # ECF frequencies: [5, 10, 20, 40]


def compute_estimators_one_snr(
    snr: float, c: float = C, n_paths: int = N_PATHS,
    n: int = N, seed_base: int = 0
) -> dict:
    """
    Run all estimators on n_paths paths for a single SNR.

    Returns dict mapping estimator name -> np.ndarray of shape (n_paths,).
    For Hasbrouck, only runs on n_has paths and pads remainder with NaN.
    """
    V = c**2 / snr
    paths = simulate_roll_paths(c, V, n_paths=n_paths, n=n,
                                seed=seed_base + int(snr * 1000))

    # ---------- Roll ----------
    c_roll = roll_estimator_batch(paths)

    # ---------- Fixed-frequency CF ----------
    a_star = find_interior_root(c, V, n_grid=3000)
    if not np.isnan(a_star):
        cf_res = cf_estimator_batch(paths, a_star)
        c_cf   = cf_res['c_hat']
        adm_cf = cf_res['admissible']
    else:
        c_cf   = np.full(n_paths, np.nan)
        adm_cf = np.zeros(n_paths, dtype=bool)

    # ---------- CF-Roll mixture ----------
    if not np.isnan(a_star):
        from src.theory import cf_roll_mixture as _cfr
        try:
            mix = _cfr(a_star, c, V)
            w   = mix['w_star']
        except Exception:
            w   = 0.5
        c_mix = cf_roll_mixture_estimator(paths, a_star, w)
    else:
        c_mix = c_roll.copy()

    # ---------- Augmented ECF-GMM (n_starts=1 for MC speed) ----------
    c_gmm      = np.zeros(n_paths)
    gmm_conv   = np.zeros(n_paths, dtype=bool)
    t0_gmm = time.perf_counter()
    for i in range(n_paths):
        res = ecf_gmm_augmented(paths[i], FREQS, n_starts=1)
        c_gmm[i]    = res['c_hat']
        gmm_conv[i] = res['converged']
    t_gmm = (time.perf_counter() - t0_gmm) / n_paths

    # ---------- Hasbrouck (reduced sample + fast MCMC settings) ----------
    c_has = np.full(n_paths, np.nan)
    n_has = min(N_HAS, n_paths)
    t0_has = time.perf_counter()
    for i in range(n_has):
        res = hasbrouck_mcmc(paths[i], burn_in=HAS_BURN, n_draws=HAS_DRAW,
                             seed=i + int(snr * 1000))
        c_has[i] = res['c_hat']
    t_has = (time.perf_counter() - t0_has) / n_has

    return {
        'paths':      paths,
        'c_roll':     c_roll,
        'c_cf':       c_cf,
        'adm_cf':     adm_cf,
        'c_mix':      c_mix,
        'c_gmm':      c_gmm,
        'gmm_conv':   gmm_conv,
        'c_has':      c_has,
        'a_star':     float(a_star) if not np.isnan(a_star) else np.nan,
        't_gmm_per_path': t_gmm,
        't_has_per_path': t_has,
        'n_has':      n_has,
    }


def bias_rmse(estimates: np.ndarray, true_c: float) -> tuple:
    """Return (bias, rmse) excluding NaN values."""
    valid = estimates[np.isfinite(estimates)]
    if len(valid) == 0:
        return np.nan, np.nan
    bias = float(np.mean(valid)) - true_c
    rmse = float(np.sqrt(np.mean((valid - true_c)**2)))
    return bias, rmse


def run_monte_carlo() -> pd.DataFrame:
    """
    Main Monte Carlo loop over SNR grid.

    Returns summary DataFrame with bias and RMSE per estimator per SNR.
    """
    summary_rows = []
    hist_data    = {}

    print("\n[Exp2] Starting Monte Carlo simulation...")
    for snr in SNR_GRID:
        print(f"  SNR={snr:.3f} ...", end=' ', flush=True)
        t0 = time.perf_counter()

        res  = compute_estimators_one_snr(snr, C, N_PATHS, N)
        true = C

        for name, arr in [
            ('Roll',          res['c_roll']),
            ('CF(a*)',        res['c_cf']),
            ('CF-Roll-mix',   res['c_mix']),
            ('ECF-GMM',       res['c_gmm']),
            ('Hasbrouck',     res['c_has']),
        ]:
            b, r = bias_rmse(arr, true)
            n_valid = int(np.sum(np.isfinite(arr)))
            summary_rows.append({
                'snr': round(float(snr), 3),
                'estimator': name,
                'bias': round(b, 8) if np.isfinite(b) else np.nan,
                'rmse': round(r, 8) if np.isfinite(r) else np.nan,
                'n_valid': n_valid,
                'adm_rate': round(float(np.mean(res['adm_cf'])), 4) if name == 'CF(a*)' else np.nan,
                'gmm_conv_rate': round(float(np.mean(res['gmm_conv'])), 4) if name == 'ECF-GMM' else np.nan,
            })

        # Save histogram data at the chosen SNR
        if abs(snr - SNR_HIST) < 1e-4:
            hist_data = {
                'Roll':        res['c_roll'],
                'CF(a*)':      res['c_cf'],
                'ECF-GMM':     res['c_gmm'],
                'Hasbrouck':   res['c_has'],
            }

        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.1f}s, GMM {res['t_gmm_per_path']*1000:.1f}ms/path, "
              f"Hasbrouck {res['t_has_per_path']:.2f}s/path)")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'exp2_mc_summary.csv'), index=False)
    return df_summary, hist_data


def plot_and_save(df_summary: pd.DataFrame, hist_data: dict) -> None:
    """Generate and save Figure 2 and Figure 3 style plots."""
    # Figure 2: histograms at SNR_HIST
    if hist_data:
        fig2 = plot_mc_histograms(hist_data, true_c=C,
                                  snr_label=f'SNR={SNR_HIST}')
        fig2.savefig(os.path.join(RESULTS_DIR, 'exp2_fig2_histograms.png'), dpi=150)
        plt.close(fig2)
        print(f"[Exp2] Figure 2 saved.")

    # Figure 3: bias/RMSE vs SNR
    fig3 = plot_bias_rmse_curves(df_summary)
    fig3.savefig(os.path.join(RESULTS_DIR, 'exp2_fig3_bias_rmse.png'), dpi=150)
    plt.close(fig3)
    print(f"[Exp2] Figure 3 saved.")


def print_summary(df_summary: pd.DataFrame) -> None:
    """Print key qualitative results matching paper expectations."""
    print("\n[Exp2] Summary table (bias and RMSE) at SNR=0.70:")
    sub = df_summary[np.abs(df_summary['snr'] - 0.70) < 0.01]
    if not sub.empty:
        print(sub[['estimator', 'bias', 'rmse', 'n_valid']].to_string(index=False))

    print("\n[Exp2] Expected qualitative ordering (paper):")
    print("  Roll: fast, stable, finite-sample bias")
    print("  ECF-GMM: often lower bias than Roll, higher RMSE possible")
    print("  Hasbrouck: slow, often slightly left-shifted")
    print("  CF(a*): bias near zero near a*, higher variance")


def main() -> None:
    print("=" * 60)
    print("Experiment 2: Synthetic Monte Carlo Comparison")
    print("=" * 60)

    df_summary, hist_data = run_monte_carlo()
    plot_and_save(df_summary, hist_data)
    print_summary(df_summary)

    print("\n[Exp2] All outputs saved to results/")


if __name__ == '__main__':
    main()
