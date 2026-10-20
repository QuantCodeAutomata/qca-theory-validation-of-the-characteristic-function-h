"""
Visualization utilities for CF estimator theory and Monte Carlo experiments.
All figures use matplotlib with Agg backend for headless rendering.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.theory import (
    cf_bias_coeff, cf_variance_coeff, transform_bias
)
from src.roots import build_frequency_grid


def plot_transformed_bias_curves(
    snr_grid: np.ndarray,
    c: float,
    n_grid: int = 5000,
    save_path: str = None,
) -> plt.Figure:
    """
    Figure 1 style: transformed bias T(b) vs frequency a for each SNR.

    T(b) = sign(b) * log10(1 + |b| / c)

    Parameters
    ----------
    snr_grid  : array-like, SNR values (e.g. 0.40, 0.425, ..., 1.00)
    c         : float, half-spread (default 0.02)
    n_grid    : int, frequency grid resolution
    save_path : str or None, file path to save (PNG)

    Returns
    -------
    matplotlib.figure.Figure
    """
    snr_grid = np.asarray(snr_grid)
    a_grid = build_frequency_grid(c, n_grid)
    a_upper = np.pi / (4.0 * c)

    cmap = cm.viridis
    colors = [cmap(i / max(len(snr_grid) - 1, 1)) for i in range(len(snr_grid))]

    fig, ax = plt.subplots(figsize=(10, 6))
    for snr, col in zip(snr_grid, colors):
        V = c**2 / snr
        b = cf_bias_coeff(a_grid, c, V)
        Tb = transform_bias(b, b0=c)
        ax.plot(a_grid, Tb, color=col, lw=0.8, label=f'SNR={snr:.3f}')

    ax.axhline(0, color='black', lw=1.2, ls='--', label='zero bias')
    ax.axvline(a_upper, color='red', lw=1.0, ls=':', label=r'$a=\pi/(4c)$')
    ax.set_xlabel('Frequency $a$', fontsize=13)
    ax.set_ylabel(r'$T(b) = \mathrm{sign}(b)\,\log_{10}(1+|b|/c)$', fontsize=12)
    ax.set_title(
        f'Transformed CF bias curves ($c={c}$, SNR grid {snr_grid[0]:.2f}-{snr_grid[-1]:.2f})',
        fontsize=13
    )
    # Add a colorbar for SNR
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(snr_grid.min(), snr_grid.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('SNR', fontsize=12)
    ax.set_xlim([a_grid[0], a_upper * 1.01])
    ax.set_ylim([-2, 2])
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_variance_tradeoff(
    c: float,
    V_list: list,
    labels: list,
    n_grid: int = 5000,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot sigma_CF^2(a;c,V) vs frequency a for several (c,V) pairs.

    Parameters
    ----------
    c         : float
    V_list    : list of float
    labels    : list of str
    n_grid    : int
    save_path : str or None
    """
    a_grid  = build_frequency_grid(c, n_grid)
    a_upper = np.pi / (4.0 * c)

    fig, ax = plt.subplots(figsize=(9, 5))
    for V, lbl in zip(V_list, labels):
        sigma2 = cf_variance_coeff(a_grid, c, V)
        ax.semilogy(a_grid, sigma2, label=lbl)

    ax.axvline(a_upper, color='red', lw=1.0, ls=':', label=r'$a=\pi/(4c)$')
    ax.set_xlabel('Frequency $a$', fontsize=13)
    ax.set_ylabel(r'$\sigma^2_{\mathrm{CF}}(a;c,V)$', fontsize=12)
    ax.set_title(f'CF variance coefficient ($c={c}$)', fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_mixture_weights(
    results_df: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot optimal mixture weight w* vs SNR.

    Parameters
    ----------
    results_df : DataFrame with columns snr, w_star
    save_path  : str or None
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(results_df['snr'], results_df['w_star'], 'o-', lw=2, ms=5)
    ax.set_xlabel('SNR', fontsize=13)
    ax.set_ylabel('Optimal mixture weight $w^*$', fontsize=12)
    ax.set_title('CF-Roll mixture optimal weight vs SNR', fontsize=13)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_mc_histograms(
    estimates_dict: dict,
    true_c: float,
    snr_label: str,
    save_path: str = None,
) -> plt.Figure:
    """
    Figure 2 style: histogram of per-path estimates for each estimator.

    Parameters
    ----------
    estimates_dict : dict mapping estimator name -> np.ndarray of estimates
    true_c         : float, true half-spread
    snr_label      : str, e.g. 'SNR=0.70'
    save_path      : str or None
    """
    n_est = len(estimates_dict)
    fig, axes = plt.subplots(1, n_est, figsize=(4 * n_est, 4), sharey=False)
    if n_est == 1:
        axes = [axes]

    for ax, (name, vals) in zip(axes, estimates_dict.items()):
        vals_clean = vals[np.isfinite(vals)]
        ax.hist(vals_clean, bins=40, edgecolor='white', alpha=0.8)
        ax.axvline(true_c, color='red', lw=2, ls='--', label=f'True c={true_c}')
        ax.axvline(float(np.mean(vals_clean)), color='navy', lw=1.5, ls='-',
                   label=f'Mean={np.mean(vals_clean):.4f}')
        ax.set_title(name, fontsize=12)
        ax.set_xlabel(r'$\hat{c}$', fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle(f'Estimator distributions ({snr_label})', fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_bias_rmse_curves(
    summary_df: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """
    Figure 3 style: bias and RMSE vs SNR for all estimators.

    Parameters
    ----------
    summary_df : DataFrame with columns snr, estimator, bias, rmse
    save_path  : str or None
    """
    estimators = summary_df['estimator'].unique()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(estimators)))
    for est, col in zip(estimators, colors):
        sub = summary_df[summary_df['estimator'] == est].sort_values('snr')
        ax1.plot(sub['snr'], sub['bias'],  'o-', lw=2, ms=5, color=col, label=est)
        ax2.plot(sub['snr'], sub['rmse'],  'o-', lw=2, ms=5, color=col, label=est)

    ax1.axhline(0, color='black', lw=1, ls='--')
    ax1.set_xlabel('SNR', fontsize=13)
    ax1.set_ylabel('Bias', fontsize=12)
    ax1.set_title('Finite-sample bias vs SNR', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('SNR', fontsize=13)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('RMSE vs SNR', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_cross_section(
    results_df: pd.DataFrame,
    year: int,
    save_path: str = None,
) -> plt.Figure:
    """
    Figure 4 style: per-stock transaction cost estimates across methods.

    Parameters
    ----------
    results_df : DataFrame with columns ticker, roll, gmm, hasbrouck
    year       : int, calendar year label
    save_path  : str or None
    """
    df = results_df.sort_values('roll').reset_index(drop=True)
    x  = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.25
    ax.bar(x - w, df['roll'],      width=w, label='Roll',   alpha=0.85)
    ax.bar(x,     df['gmm'],       width=w, label='ECF-GMM', alpha=0.85)
    ax.bar(x + w, df['hasbrouck'], width=w, label='Hasbrouck', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df['ticker'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Transaction cost (log basis points)', fontsize=12)
    ax.set_title(
        f'Cross-sectional transaction cost estimates {year} (SYNTHETIC DATA)',
        fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
