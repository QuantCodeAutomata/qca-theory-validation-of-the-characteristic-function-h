"""
Experiment 4: CPU runtime benchmarking of transaction-cost estimators.

Setup:
  1000 synthetic paths, n=252, c=0.02
  SNR ∈ {0.40, 0.70, 1.00}

Estimators benchmarked:
  1. Roll       (closed-form)
  2. Fixed-freq CF at a*(c,V)
  3. Augmented ECF-GMM (two-step feasible GMM)
  4. Hasbrouck  (Gibbs MCMC, burn_in=1000, n_draws=1500)

Paper reported timings (Intel Xeon Gold 6530, CPU only):
  Roll:        0.000276 s/path
  ECF-GMM:     0.005769 s/path
  Hasbrouck:  39.525235 s/path

NOTE: All estimators are run serially (single process) to ensure fair comparison.
Hasbrouck is timed on all 1000 paths (this will take significant time ~30-40k sec
on real hardware; for practical purposes we reduce to 50 paths and extrapolate,
documenting this as a hardware-constrained reconstruction).
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
from src.roll import roll_estimator
from src.cf import cf_estimator_single
from src.gmm import ecf_gmm_augmented, DEFAULT_FREQS
from src.hasbrouck import hasbrouck_mcmc
from src.roots import find_interior_root
from src.runtime import get_system_info, time_estimator, summarize_timings

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

C       = 0.02
N       = 252
N_PATHS = 1000
N_HAS   = 50      # Hasbrouck subsample for speed
SNR_LIST = [0.40, 0.70, 1.00]
FREQS    = DEFAULT_FREQS

PAPER_TIMES = {
    'Roll':        0.000276,
    'ECF-GMM':     0.005769,
    'Hasbrouck':  39.525235,
}


def make_roll_fn(Y: np.ndarray) -> None:
    roll_estimator(Y)


def make_cf_fn(a_star: float):
    def fn(Y: np.ndarray) -> None:
        cf_estimator_single(Y, a_star)
    return fn


def make_gmm_fn(Y: np.ndarray) -> None:
    ecf_gmm_augmented(Y, FREQS, n_starts=3)


def make_hasbrouck_fn(Y: np.ndarray) -> None:
    hasbrouck_mcmc(Y, burn_in=1000, n_draws=1500)


def benchmark_snr(snr: float) -> dict:
    """
    Benchmark all estimators on n_paths paths at given SNR.

    Returns dict of estimator -> timing summary dict.
    """
    V = C**2 / snr
    a_star = find_interior_root(C, V, n_grid=3000)

    # Generate paths (excluded from timing)
    paths = simulate_roll_paths(C, V, n_paths=N_PATHS, n=N,
                                seed=int(snr * 1000))

    results = {}

    # ---- Warm-up (avoids cold-start timing artifacts) ----
    for _ in range(3):
        roll_estimator(paths[0])
        if not np.isnan(a_star):
            cf_estimator_single(paths[0], a_star)
        ecf_gmm_augmented(paths[0], FREQS, n_starts=1)

    # ---- Roll ----
    print(f"  SNR={snr:.2f}: timing Roll ({N_PATHS} paths)...", end=' ', flush=True)
    t0 = time.perf_counter()
    for Y in paths:
        roll_estimator(Y)
    t_roll = (time.perf_counter() - t0) / N_PATHS
    results['Roll'] = t_roll
    print(f"{t_roll*1000:.4f} ms/path")

    # ---- Fixed-freq CF ----
    if not np.isnan(a_star):
        print(f"  SNR={snr:.2f}: timing CF(a*) ({N_PATHS} paths)...", end=' ', flush=True)
        t0 = time.perf_counter()
        for Y in paths:
            cf_estimator_single(Y, a_star)
        t_cf = (time.perf_counter() - t0) / N_PATHS
        results['CF(a*)'] = t_cf
        print(f"{t_cf*1000:.4f} ms/path")

    # ---- Augmented ECF-GMM ----
    print(f"  SNR={snr:.2f}: timing ECF-GMM ({N_PATHS} paths)...", end=' ', flush=True)
    t0 = time.perf_counter()
    for Y in paths:
        ecf_gmm_augmented(Y, FREQS, n_starts=3)
    t_gmm = (time.perf_counter() - t0) / N_PATHS
    results['ECF-GMM'] = t_gmm
    print(f"{t_gmm*1000:.4f} ms/path")

    # ---- Hasbrouck (reduced sample, extrapolated) ----
    print(f"  SNR={snr:.2f}: timing Hasbrouck ({N_HAS} paths, extrapolated to {N_PATHS})...",
          end=' ', flush=True)
    t0 = time.perf_counter()
    for Y in paths[:N_HAS]:
        hasbrouck_mcmc(Y, burn_in=1000, n_draws=1500, seed=42)
    t_has = (time.perf_counter() - t0) / N_HAS
    results['Hasbrouck'] = t_has
    print(f"{t_has:.4f} s/path (extrap.)")

    return results


def build_timing_table(all_results: dict) -> pd.DataFrame:
    """
    Summarize timing results across SNR values.

    all_results : dict mapping snr -> dict of estimator -> mean_time_sec
    """
    rows = []
    for snr, est_times in all_results.items():
        for est, t in est_times.items():
            paper_t = PAPER_TIMES.get(est, None)
            rows.append({
                'snr':            round(float(snr), 2),
                'estimator':      est,
                'mean_sec':       round(t, 8),
                'mean_ms':        round(t * 1000, 6),
                'paper_sec':      paper_t,
                'ratio_vs_paper': round(t / paper_t, 3) if paper_t else None,
            })
    return pd.DataFrame(rows)


def print_speed_ratios(df: pd.DataFrame) -> None:
    """Print speed comparison ratios (key paper claim)."""
    print("\n[Exp4] Speed ratios at each SNR:")
    for snr in SNR_LIST:
        sub = df[df['snr'] == snr].set_index('estimator')
        if 'Roll' not in sub.index or 'ECF-GMM' not in sub.index:
            continue
        t_roll = sub.loc['Roll', 'mean_sec']
        t_gmm  = sub.loc['ECF-GMM', 'mean_sec']
        if 'Hasbrouck' in sub.index:
            t_has = sub.loc['Hasbrouck', 'mean_sec']
            print(f"  SNR={snr:.2f}: Hasbrouck/GMM={t_has/t_gmm:.0f}x  "
                  f"GMM/Roll={t_gmm/t_roll:.0f}x  "
                  f"Hasbrouck/Roll={t_has/t_roll:.0f}x")

    print("\n[Exp4] Paper reported (Intel Xeon Gold 6530):")
    print(f"  Roll:      {PAPER_TIMES['Roll']*1000:.3f} ms/path")
    print(f"  ECF-GMM:   {PAPER_TIMES['ECF-GMM']*1000:.3f} ms/path")
    print(f"  Hasbrouck: {PAPER_TIMES['Hasbrouck']:.3f} s/path")
    print(f"  Paper Hasbrouck/GMM ratio: {PAPER_TIMES['Hasbrouck']/PAPER_TIMES['ECF-GMM']:.0f}x")


def plot_timing_bars(df: pd.DataFrame) -> None:
    """Bar plot of mean runtime per estimator (log scale)."""
    # Average across SNR values
    avg = df.groupby('estimator')['mean_sec'].mean().reset_index()
    avg = avg.sort_values('mean_sec')

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(avg['estimator'], avg['mean_sec'] * 1000, log=True, alpha=0.85)
    ax.set_xlabel('Estimator', fontsize=13)
    ax.set_ylabel('Mean runtime per path (ms, log scale)', fontsize=12)
    ax.set_title('Runtime benchmark: CF-based estimators (avg over SNR grid)', fontsize=13)
    ax.grid(True, axis='y', which='both', alpha=0.3)

    # Label bars with paper values
    for bar, est in zip(bars, avg['estimator']):
        paper_t = PAPER_TIMES.get(est)
        if paper_t:
            ax.axhline(paper_t * 1000, color='red', lw=1.0, ls='--', alpha=0.6)
    ax.annotate('---- Paper values (red dashes)', xy=(0.6, 0.92),
                xycoords='axes fraction', color='red', fontsize=9)

    fig.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'exp4_runtime_bars.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Exp4] Runtime bar chart saved to {save_path}")


def main() -> None:
    print("=" * 60)
    print("Experiment 4: CPU Runtime Benchmarking")
    print("=" * 60)

    # System info
    sysinfo = get_system_info()
    print("\n[Exp4] System info:")
    for k, v in sysinfo.items():
        print(f"  {k}: {v}")

    sysinfo_df = pd.DataFrame([sysinfo])
    sysinfo_df.to_csv(os.path.join(RESULTS_DIR, 'exp4_sysinfo.csv'), index=False)

    # Benchmark
    all_results = {}
    for snr in SNR_LIST:
        print(f"\n[Exp4] Benchmarking SNR={snr:.2f}...")
        all_results[snr] = benchmark_snr(snr)

    df_timing = build_timing_table(all_results)
    df_timing.to_csv(os.path.join(RESULTS_DIR, 'exp4_timing.csv'), index=False)

    print("\n[Exp4] Timing table:")
    print(df_timing.to_string(index=False))

    print_speed_ratios(df_timing)
    plot_timing_bars(df_timing)

    print("\n[Exp4] Expected ranking: Roll < CF(a*) < ECF-GMM << Hasbrouck")
    print("[Exp4] All outputs saved to results/")


if __name__ == '__main__':
    main()
