"""Best-lag regression with inference that accounts for the lag search."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_LAGS = tuple(range(-6, 7))
DEFAULT_BLOCK = 24
DEFAULT_N_PERM = 1999


def _lagged(x: np.ndarray, y: np.ndarray, lag: int):
    """Same pairing convention as composite_stats.best_lag_slope."""
    if lag < 0:
        xl = x[-lag:]
        yl = y[:len(xl)]
    elif lag > 0:
        xl, yl = x[:-lag], y[lag:]
    else:
        xl, yl = x, y
    return xl, yl


def best_lag_stats(x, y, lags=DEFAULT_LAGS, min_n: int = 24):
    """The lag with the highest R², as {lag, slope, r2, n}; None if no lag has >= min_n pairs."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    best = None
    for lag in lags:
        xl, yl = _lagged(x, y, int(lag))
        ok = np.isfinite(xl) & np.isfinite(yl)
        n = int(ok.sum())
        if n < min_n:
            continue
        xm, ym = xl[ok] - xl[ok].mean(), yl[ok] - yl[ok].mean()
        sxx, syy, sxy = float(xm @ xm), float(ym @ ym), float(xm @ ym)
        if sxx <= 0 or syy <= 0:
            continue
        r2 = sxy * sxy / (sxx * syy)
        if best is None or r2 > best["r2"]:
            best = {"lag": int(lag), "slope": sxy / sxx, "r2": r2, "n": n}
    return best


def block_permute(x: np.ndarray, block: int, rng) -> np.ndarray:
    """Cut into consecutive blocks, permute the blocks, keep the original length."""
    n = len(x)
    k = int(np.ceil(n / block))
    blocks = [x[i * block:(i + 1) * block] for i in range(k)]
    order = rng.permutation(k)
    return np.concatenate([blocks[i] for i in order])[:n]


def best_lag_randomization(x, y, lags=DEFAULT_LAGS, min_n: int = 24, block: int = DEFAULT_BLOCK,
                           n_perm: int = DEFAULT_N_PERM, seed: int = 42):
    """Best-lag regression whose p-value accounts for the lag search and for autocorrelation."""
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    x = x.reindex(y.index)
    ok = x.notna() & y.notna()
    xa, ya = x[ok].values, y[ok].values

    obs = best_lag_stats(xa, ya, lags, min_n)
    if obs is None:
        return None

    rng = np.random.default_rng(seed)
    null = np.empty(int(n_perm))
    for i in range(int(n_perm)):
        s = best_lag_stats(block_permute(xa, block, rng), ya, lags, min_n)
        null[i] = s["r2"] if s is not None else np.nan
    null = null[np.isfinite(null)]

    t = np.sqrt(obs["r2"] * (obs["n"] - 2) / max(1.0 - obs["r2"], 1e-12))
    return {**obs, "p_value": float((1 + np.sum(null >= obs["r2"])) / (len(null) + 1)),
            "p_ols_naive": float(2.0 * stats.t.sf(t, obs["n"] - 2)),
            "n_perm": int(len(null)), "block": int(block)}
