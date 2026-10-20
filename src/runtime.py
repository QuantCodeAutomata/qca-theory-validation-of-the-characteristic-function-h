"""
Runtime benchmarking utilities for transaction-cost estimators.

Uses time.perf_counter for high-resolution monotonic timing.
Records system metadata for reproducibility.
"""

import os
import time
import platform

import numpy as np
import pandas as pd

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def get_system_info() -> dict:
    """
    Collect system metadata for reproducibility reporting.

    Returns
    -------
    dict with platform, processor, cpu_count, package versions, etc.
    """
    info = {
        'python_version':  platform.python_version(),
        'platform':        platform.platform(),
        'processor':       platform.processor(),
        'cpu_count_logical': os.cpu_count(),
        'numpy_version':   np.__version__,
        'pandas_version':  pd.__version__,
    }
    if _HAS_PSUTIL:
        freq = psutil.cpu_freq()
        info['cpu_freq_mhz'] = freq.current if freq else None
        info['ram_gb']       = round(psutil.virtual_memory().total / 1e9, 2)
    return info


def time_estimator(
    func, paths: np.ndarray, **kwargs
) -> pd.DataFrame:
    """
    Time an estimator function path-by-path.

    Parameters
    ----------
    func   : callable(Y: np.ndarray, **kwargs) -> any
    paths  : np.ndarray of shape (n_paths, n)
    kwargs : additional keyword arguments forwarded to func

    Returns
    -------
    pd.DataFrame with columns: path_id, elapsed_seconds
    """
    records = []
    for i, Y in enumerate(paths):
        t0 = time.perf_counter()
        func(Y, **kwargs)
        elapsed = time.perf_counter() - t0
        records.append({'path_id': i, 'elapsed_seconds': elapsed})
    return pd.DataFrame(records)


def summarize_timings(timing_df: pd.DataFrame) -> dict:
    """
    Compute summary statistics of elapsed time (seconds).

    Parameters
    ----------
    timing_df : DataFrame with column 'elapsed_seconds'

    Returns
    -------
    dict with keys: mean, median, std, min, max, p95
    """
    t = timing_df['elapsed_seconds']
    return {
        'mean':   float(t.mean()),
        'median': float(t.median()),
        'std':    float(t.std()),
        'min':    float(t.min()),
        'max':    float(t.max()),
        'p95':    float(t.quantile(0.95)),
    }
