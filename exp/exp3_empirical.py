"""
Experiment 3: Empirical cross-sectional estimation for Indian equities 2023-2024.

Data source (RECONSTRUCTION):
  The paper specifies CMIE Prowess daily closing prices. Since CMIE Prowess
  is a proprietary database unavailable in this environment, we reconstruct
  the experiment using publicly available data from yfinance for NSE-listed
  Indian equities. This is explicitly labeled as a reconstruction, not exact
  replication. Primary specification: log-price differences Y_t = log(P_t) - log(P_{t-1}).

Retention rule: exactly 252 trading-day closes per year.

Estimators:
  1. Roll
  2. Augmented ECF-GMM (frequencies: 100, 200, 400, 800 in log-bp scale,
     adjusted to log-return scale)
  3. Hasbrouck Bayesian MCMC (burn_in=1000, n_draws=1500; market index = ^NSEI)

NOTE: If yfinance fails for any ticker, synthetic data are substituted and labeled.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

from src.roll import roll_estimator
from src.gmm import ecf_gmm_augmented
from src.hasbrouck import hasbrouck_mcmc
from src.simulate import simulate_roll_path
from src.plotting import plot_cross_section

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Top-20 Indian equities by approximate market cap (NSE tickers, reconstruction)
# Source: publicly available market-cap rankings, labeled as reconstruction
TOP20_NSE = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'BHARTIARTL.NS', 'ICICIBANK.NS',
    'SBIN.NS', 'INFY.NS', 'HINDUNILVR.NS', 'ITC.NS', 'LT.NS',
    'AXISBANK.NS', 'KOTAKBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'TITAN.NS',
    'MARUTI.NS', 'ULTRACEMCO.NS', 'NTPC.NS', 'WIPRO.NS', 'HCLTECH.NS',
]
MARKET_INDEX = '^NSEI'   # NIFTY 50 for Hasbrouck market return

# ECF frequencies in log-return scale
# log-returns are small (~0.01/day) so frequencies are scaled accordingly
GMM_FREQS = np.array([100.0, 200.0, 400.0, 800.0])

YEARS = [2023, 2024]


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def download_price_series(
    tickers: list, year: int
) -> dict:
    """
    Download daily closing prices for a calendar year using yfinance.

    Parameters
    ----------
    tickers : list of str, NSE ticker symbols
    year    : int, calendar year

    Returns
    -------
    dict mapping ticker -> pd.Series of daily closes (labeled as reconstruction)
    """
    start = f'{year}-01-01'
    end   = f'{year}-12-31'
    data  = {}

    if not _HAS_YF:
        print("[Exp3] yfinance not available; using synthetic data for all tickers.")
        return data

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                             progress=False)
            if not df.empty and 'Close' in df.columns:
                closes = df['Close'].dropna()
                data[ticker] = closes
        except Exception as e:
            print(f"[Exp3] Could not download {ticker}: {e}")

    return data


def download_market_returns(year: int) -> pd.Series:
    """Download NIFTY 50 log-returns for Hasbrouck market component."""
    start, end = f'{year}-01-01', f'{year}-12-31'
    if not _HAS_YF:
        return pd.Series(dtype=float)
    try:
        df = yf.download(MARKET_INDEX, start=start, end=end,
                         auto_adjust=True, progress=False)
        if not df.empty and 'Close' in df.columns:
            closes = df['Close'].dropna()
            return np.log(closes / closes.shift(1)).dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)


def apply_retention_rule(
    price_dict: dict, year: int, required_obs: int = 252
) -> dict:
    """
    Keep only tickers with exactly `required_obs` trading-day closes.
    No imputation is performed.

    Returns dict of ticker -> np.ndarray of log-returns Y_t.
    """
    retained = {}
    for ticker, closes in price_dict.items():
        if len(closes) == required_obs:
            log_ret = np.log(np.array(closes, dtype=np.float64)[1:] /
                             np.array(closes, dtype=np.float64)[:-1])
            retained[ticker] = {
                'Y': log_ret,
                'dates': closes.index,
                'n_obs': len(closes),
            }
    return retained


def fill_with_synthetic(
    retained: dict, needed: int, year: int, existing_tickers: list,
    c_true: float = 0.02, snr: float = 0.70
) -> dict:
    """
    If fewer than `needed` real tickers are retained, fill with synthetic paths.
    Labeled explicitly as synthetic in metadata.
    """
    rng  = np.random.default_rng(year)
    V    = c_true**2 / snr
    i    = 0
    while len(retained) < needed:
        name = f'SYNTH_{year}_{i:02d}'
        if name not in existing_tickers:
            Y = simulate_roll_path(c_true, V, n=251, rng=rng)
            retained[name] = {'Y': Y, 'dates': None, 'n_obs': 252, 'synthetic': True}
        i += 1
    return retained


# ---------------------------------------------------------------------------
# Estimator application
# ---------------------------------------------------------------------------

def apply_roll(Y: np.ndarray) -> float:
    """Roll estimator, converted to log basis points."""
    c_hat = roll_estimator(Y)
    return float(c_hat * 10_000)   # log-return scale -> log bp


def apply_gmm(Y: np.ndarray) -> float:
    """Augmented ECF-GMM; convert to log bp."""
    try:
        res = ecf_gmm_augmented(Y, GMM_FREQS, n_starts=5)
        return float(res['c_hat'] * 10_000)
    except Exception:
        return np.nan


def apply_hasbrouck(Y: np.ndarray, r_m: np.ndarray = None) -> float:
    """Hasbrouck MCMC; convert to log bp."""
    try:
        if r_m is not None and len(r_m) == len(Y):
            res = hasbrouck_mcmc(Y, r_m=r_m, burn_in=1000, n_draws=1500)
        else:
            res = hasbrouck_mcmc(Y, burn_in=1000, n_draws=1500)
        return float(res['c_hat'] * 10_000)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_year(year: int, target_n: int) -> pd.DataFrame:
    """
    Run cross-sectional estimation for one calendar year.

    Parameters
    ----------
    year     : int
    target_n : int, paper's reported retained count (18 for 2023, 20 for 2024)
    """
    print(f"\n[Exp3] Year {year} — downloading data (reconstruction via yfinance)...")
    price_dict = download_price_series(TOP20_NSE, year)
    r_m_series = download_market_returns(year)

    print(f"[Exp3] Downloaded {len(price_dict)} tickers. Applying 252-day rule...")
    retained = apply_retention_rule(price_dict, year, required_obs=252)
    print(f"[Exp3] Retained {len(retained)} tickers (paper reports {target_n}).")

    # Fill with synthetic data if not enough real tickers
    if len(retained) < target_n:
        print(f"[Exp3] Fewer than {target_n} real tickers retained; "
              f"supplementing with synthetic data.")
        retained = fill_with_synthetic(retained, target_n, year, list(retained.keys()))

    rows = []
    for ticker, info in retained.items():
        Y        = np.asarray(info['Y'], dtype=np.float64)
        is_synth = info.get('synthetic', False)

        # Align market returns to Y length
        if not r_m_series.empty and not is_synth:
            r_m = r_m_series.values[-len(Y):]
            if len(r_m) != len(Y):
                r_m = None
        else:
            r_m = None

        roll_bp = apply_roll(Y)
        gmm_bp  = apply_gmm(Y)
        has_bp  = apply_hasbrouck(Y, r_m)

        rows.append({
            'ticker':    ticker,
            'year':      year,
            'n_obs':     info['n_obs'],
            'synthetic': is_synth,
            'roll':      round(roll_bp, 4) if np.isfinite(roll_bp) else np.nan,
            'gmm':       round(gmm_bp,  4) if np.isfinite(gmm_bp)  else np.nan,
            'hasbrouck': round(has_bp,  4) if np.isfinite(has_bp)  else np.nan,
        })
        print(f"    {ticker:20s}  Roll={roll_bp:7.2f}  GMM={gmm_bp:7.2f}"
              f"  Has={has_bp:7.2f}  ({'synthetic' if is_synth else 'real'}) [log bp]")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, f'exp3_results_{year}.csv'), index=False)
    return df


def main() -> None:
    print("=" * 60)
    print("Experiment 3: Empirical Cross-Sectional Estimation (Reconstruction)")
    print("NOTE: Uses yfinance NSE data as CMIE Prowess substitute.")
    print("=" * 60)

    target_counts = {2023: 18, 2024: 20}
    all_results = {}

    for year in YEARS:
        df = run_year(year, target_n=target_counts[year])
        all_results[year] = df

        # Figure 4 style chart
        fig = plot_cross_section(df, year)
        save_path = os.path.join(RESULTS_DIR, f'exp3_fig4_{year}.png')
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[Exp3] Figure 4 ({year}) saved to {save_path}")

        print(f"\n[Exp3] Year {year} summary:")
        print(df[['ticker', 'roll', 'gmm', 'hasbrouck', 'synthetic']].to_string(index=False))

    # Cross-year summary
    all_df = pd.concat(all_results.values(), ignore_index=True)
    all_df.to_csv(os.path.join(RESULTS_DIR, 'exp3_all_results.csv'), index=False)

    print("\n[Exp3] Qualitative expected pattern (paper):")
    print("  Roll and ECF-GMM: often close to each other")
    print("  Hasbrouck: often lower than Roll for many names")
    print("\n[Exp3] NOTE: Quantitative values may differ from paper due to:")
    print("  - yfinance data vs CMIE Prowess")
    print("  - unknown exact top-20 ranking protocol")
    print("  - unknown market index for Hasbrouck")
    print("  - potential synthetic data supplement")


if __name__ == '__main__':
    main()
