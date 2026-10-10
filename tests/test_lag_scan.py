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


def test_a_true_lagged_link_is_recovered_at_the_right_lag_with_a_small_p():
    x, y = _pair(link=0.6, lag=2, seed=1)
    r = best_lag_randomization(x, y, n_perm=299)
    assert r["lag"] == 2 and r["p_value"] < 0.01
    assert r["p_value"] >= 1 / 300


def test_block_permutation_keeps_every_value_and_within_block_order():
    rng = np.random.default_rng(0)
    x = np.arange(100.0)
    xp = block_permute(x, 24, rng)
    assert sorted(xp) == sorted(x) and len(xp) == 100
    starts = [i for i in range(0, 96, 24)]
    assert any(np.array_equal(xp[i:i + 24], x[j:j + 24]) for i in starts[:1] for j in (0, 24, 48, 72))
