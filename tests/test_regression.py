import numpy as np
import pandas as pd
import pytest

from componergy.analysis.regression import (
    design_matrix,
    newey_west_cov,
    ols_hac,
    event_effects,
    amplification_contrast,
    amplification_table,
    adjusted_composite_table,
    normal_baseline_anomalies,
    baseline_adjusted_z,
    randomization_inference,
    shift_range,
    shift_null_coefficients,
    year_fraction,
)


def _series(n=204, seed=0, seasonal_amp=10.0, trend_per_decade=0.0, ar=0.5, noise=1.0, start="2008-01-01"):
    idx = pd.date_range(start, periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    season = seasonal_amp * np.sin(2 * np.pi * (idx.month.values - 4) / 12)
    yr = year_fraction(idx)
    trend = trend_per_decade * (yr - yr.mean()) / 10.0
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = ar * e[i - 1] + rng.normal(0, noise)
    return pd.Series(season + trend + e, index=idx)


def _mask(idx, months_by_year):
    keys = {(y, m) for y, ms in months_by_year.items() for m in ms}
    return pd.Series([(t.year, t.month) in keys for t in idx], index=idx)


def _three_masks(idx):
    m = {
        "Heatwave": _mask(idx, {2009: [6, 7, 8], 2012: [7, 8], 2015: [6, 7, 8], 2018: [7, 8]}),
        "Drought": _mask(idx, {2010: [5, 6, 7], 2013: [6, 7], 2016: [5, 6, 7], 2019: [6, 7]}),
        "Compound": _mask(idx, {2011: [7, 8, 9], 2014: [7, 8], 2017: [7, 8, 9], 2021: [6, 7, 8]}),
    }
    return m, ~(m["Heatwave"] | m["Drought"] | m["Compound"])


@pytest.mark.parametrize("lags", [0, 4, 12])
def test_newey_west_matches_statsmodels_to_machine_precision(lags):
    sm = pytest.importorskip("statsmodels.api")
    y = _series(seed=3, ar=0.7)
    ev = _mask(y.index, {2012: [6, 7, 8], 2013: [7, 8], 2019: [5, 6, 7, 8]})
    X, names = design_matrix(y.index, {"ev": ev.values}, trend="linear")
    mine = ols_hac(y.values, X, names, lags=lags, use_t=True)
    ref = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}, use_t=True)
    np.testing.assert_allclose(mine["beta"], ref.params, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(mine["cov"], ref.cov_params(), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(mine["p"], ref.pvalues, rtol=1e-6, atol=1e-12)


def test_design_matrix_rejects_unknown_trend():
    with pytest.raises(ValueError):
        design_matrix(pd.date_range("2010-01-01", periods=24, freq="MS"), trend="cubic")
