import numpy as np
import pandas as pd
import pytest

from componergy.analysis.lag_scan import best_lag_stats, best_lag_randomization, block_permute
from componergy.figures.composite_stats import best_lag_slope


def _ar1(n, phi, rng):
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + rng.normal()
    return e


def _pair(n=204, phi=0.6, link=0.0, lag=2, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-01", periods=n, freq="MS")
    x = _ar1(n, phi, rng)
    y = _ar1(n, phi, rng) + link * np.roll(x, lag)
    return pd.Series(x, index=idx), pd.Series(y, index=idx)


def test_observed_statistic_matches_the_existing_best_lag_slope_exactly():
    x, y = _pair(link=0.5, seed=3)
    mine = best_lag_stats(x.values, y.values, range(-6, 7), 24)
    ref = best_lag_slope(x, y, lags=range(-6, 7), min_n=24)
    assert mine["lag"] == ref["lag"] and mine["n"] == ref["n"]
    assert mine["slope"] == pytest.approx(ref["slope"], rel=1e-9)
    assert mine["r2"] == pytest.approx(ref["r2"], rel=1e-9)
