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


def test_seasonal_confound_is_not_reported_as_an_event_effect():
    y = _series(seed=1, seasonal_amp=10.0, ar=0.3)
    idx = y.index
    masks = {"Heatwave": _mask(idx, {yy: [7] for yy in range(2008, 2014)}),
             "Drought": _mask(idx, {yy: [8] for yy in range(2014, 2020)}),
             "Compound": _mask(idx, {yy: [7, 8] for yy in range(2020, 2025)})}
    normal = ~(masks["Heatwave"] | masks["Drought"] | masks["Compound"])

    naive = y[masks["Compound"]].mean() - y[normal].mean()
    assert naive > 5.0

    eff = event_effects(y, masks, normal, trend="linear")["effects"].set_index("Event")
    for ev in ("Heatwave", "Drought", "Compound"):
        assert abs(eff.loc[ev, "Delta_native"]) < 1.5
        assert eff.loc[ev, "CI_lo_native"] < 0 < eff.loc[ev, "CI_hi_native"]


def test_trend_confound_is_removed_only_when_a_trend_is_modelled():
    y = _series(seed=2, seasonal_amp=0.0, trend_per_decade=5.0, ar=0.2)
    idx = y.index
    late = _mask(idx, {yy: list(range(1, 13)) for yy in (2022, 2023, 2024)})
    masks = {"Heatwave": late, "Drought": pd.Series(False, index=idx), "Compound": pd.Series(False, index=idx)}
    no_trend = event_effects(y, masks, ~late, trend="none", month_fe=False)["effects"].set_index("Event")
    with_trend = event_effects(y, masks, ~late, trend="linear", month_fe=False)["effects"].set_index("Event")
    assert no_trend.loc["Heatwave", "Delta_native"] > 3.0
    assert abs(with_trend.loc["Heatwave", "Delta_native"]) < 1.0
