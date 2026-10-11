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


def test_effects_and_amplification_contrast_are_recovered():
    y = _series(seed=4, seasonal_amp=8.0, ar=0.4)
    masks, normal = _three_masks(y.index)
    y = y.copy()
    y[masks["Heatwave"]] += 2.0
    y[masks["Drought"]] += 1.0
    y[masks["Compound"]] += 6.0

    res = event_effects(y, masks, normal, trend="linear")
    eff = res["effects"].set_index("Event")
    for ev, truth in {"Heatwave": 2.0, "Drought": 1.0, "Compound": 6.0}.items():
        assert eff.loc[ev, "CI_lo_native"] < truth < eff.loc[ev, "CI_hi_native"]

    c = amplification_contrast(res)
    assert c["ci_lo"] < 3.0 < c["ci_hi"]
    assert c["p_value"] < 0.05 and c["p_exact"] < 0.05
    assert abs(c["additive_expectation"] - 3.0) < 1.5


def test_amplification_contrast_is_null_when_effects_are_purely_additive():
    y = _series(seed=5, seasonal_amp=8.0, ar=0.4)
    masks, normal = _three_masks(y.index)
    y = y.copy()
    y[masks["Heatwave"]] += 2.0
    y[masks["Drought"]] += 1.0
    y[masks["Compound"]] += 3.0
    c = amplification_contrast(event_effects(y, masks, normal))
    assert c["ci_lo"] < 0 < c["ci_hi"]
    assert c["p_value"] > 0.05


def test_amplification_contrast_is_nan_when_a_category_is_too_small():
    y = _series(seed=6)
    idx = y.index
    masks = {"Heatwave": _mask(idx, {2010: [7]}), "Drought": _mask(idx, {2012: [6, 7, 8]}),
             "Compound": _mask(idx, {2014: [6, 7, 8]})}
    normal = ~(masks["Heatwave"] | masks["Drought"] | masks["Compound"])
    res = event_effects(y, masks, normal)
    assert np.isnan(amplification_contrast(res)["contrast"])
    eff = res["effects"].set_index("Event")
    assert np.isnan(eff.loc["Heatwave", "Delta_native"]) and eff.loc["Heatwave", "N_event"] == 1


def test_months_that_are_neither_normal_nor_event_are_dropped_and_counted():
    y = _series(seed=7)
    idx = y.index
    masks = {"Heatwave": _mask(idx, {2010: [6, 7, 8]}), "Drought": _mask(idx, {2012: [6, 7, 8]}),
             "Compound": _mask(idx, {2014: [6, 7, 8]})}
    other = _mask(idx, {2016: [6, 7, 8, 9]})
    normal = ~(masks["Heatwave"] | masks["Drought"] | masks["Compound"] | other)
    assert event_effects(y, masks, normal)["info"]["n_dropped_other"] == 4


def test_year_fixed_effects_survive_an_event_that_fills_a_whole_year():
    y = _series(seed=8, seasonal_amp=3.0)
    idx = y.index
    masks = {"Heatwave": _mask(idx, {2015: list(range(1, 13))}), "Drought": _mask(idx, {2012: [6, 7, 8]}),
             "Compound": _mask(idx, {2018: [6, 7, 8]})}
    normal = ~(masks["Heatwave"] | masks["Drought"] | masks["Compound"])
    res = event_effects(y, masks, normal, trend="year_fe")
    assert res["info"]["rank_deficient"] is True


def test_shift_range_excludes_shifts_that_would_overlap_the_original_calendar():
    s = shift_range(204, 12)
    assert s.min() == 12 and s.max() == 204 - 12 and len(s) == 181
    assert len(shift_range(10, 12)) >= 1


def test_randomization_inference_reports_the_floor_of_the_exact_p_value():
    null = np.random.default_rng(0).normal(0, 1, 181)
    ri = randomization_inference(estimate=50.0, null=null)
    assert ri["p_exact"] == pytest.approx(1 / 182)
    assert ri["p_value"] < 1e-10
    assert ri["n_shifts"] == 181


def test_shift_null_matches_direct_refit_for_a_single_shift():
    y = _series(seed=9, seasonal_amp=6.0)
    masks, normal = _three_masks(y.index)
    Z, _ = design_matrix(y.index, None, trend="linear")
    D = np.column_stack([masks[k].values.astype(float) for k in ("Heatwave", "Drought", "Compound")])
    null = shift_null_coefficients(y.values, Z, D, min_shift=12)

    s = shift_range(len(y), 12)[7]
    Xs = np.column_stack([Z, np.roll(D, int(s), axis=0)])
    direct, *_ = np.linalg.lstsq(Xs, y.values, rcond=None)
    np.testing.assert_allclose(null[7], direct[-3:], rtol=1e-8, atol=1e-9)


def test_randomization_test_is_calibrated_under_the_null_with_clustered_events():
    n_sims, rej_exact, rej_hac = 120, 0, 0
    rng = np.random.default_rng(123)
    idx = pd.date_range("2008-01-01", periods=204, freq="MS")
    for _ in range(n_sims):
        e = np.zeros(204)
        for i in range(1, 204):
            e[i] = 0.6 * e[i - 1] + rng.normal()
        starts = rng.choice(np.arange(0, 190, 14), size=2, replace=False)
        ev = np.zeros(204, bool)
        for s in starts:
            ev[s:s + 8] = True
        masks = {"Heatwave": pd.Series(ev, index=idx), "Drought": pd.Series(False, index=idx),
                 "Compound": pd.Series(False, index=idx)}
        r = event_effects(pd.Series(e, index=idx), masks, pd.Series(~ev, index=idx), trend="linear",
                          month_fe=True)["effects"].set_index("Event").loc["Heatwave"]
        rej_exact += r["p_exact"] < 0.05
        rej_hac += r["p_hac"] < 0.05
    assert rej_exact / n_sims < 0.13
    assert rej_hac / n_sims > rej_exact / n_sims


def test_normal_baseline_anomalies_average_zero_in_normal_months_and_recover_the_effect():
    y = _series(seed=10, seasonal_amp=8.0, ar=0.3)
    masks, normal = _three_masks(y.index)
    ev = masks["Compound"]
    y = y.copy()
    y[ev] += 4.0
    anom, base = normal_baseline_anomalies(y, normal | masks["Heatwave"] | masks["Drought"] & ~ev)
    assert 3.0 < anom[ev].mean() < 5.0
    assert base.index.equals(y.index)


def test_baseline_adjusted_z_has_unit_standard_deviation_per_column():
    y = pd.DataFrame({"a": _series(seed=1), "b": 5 * _series(seed=2)})
    z = baseline_adjusted_z(y, pd.Series(True, index=y.index))
    np.testing.assert_allclose(z.std(ddof=1).values, [1.0, 1.0], rtol=1e-9)


def test_adjusted_composite_table_keeps_the_schema_the_figures_read():
    y = pd.DataFrame({"R1": _series(seed=11), "R2": _series(seed=12)})
    masks, normal = _three_masks(y.index)
    t = adjusted_composite_table(y, masks, normal)
    need = {"Event", "Source", "N_event", "N_normal", "Mean_event", "Mean_normal", "Difference", "tstat",
            "p_value", "Delta_native", "Delta_pct", "Delta_z", "CI_lo_native", "CI_hi_native",
            "p_exact", "p_hac", "Scale"}
    assert need <= set(t.columns)
    assert len(t) == 6 and set(t["Source"]) == {"R1", "R2"}
    np.testing.assert_allclose(t["Difference"], t["Delta_z"])
    np.testing.assert_allclose(t["Mean_event"] - t["Mean_normal"], t["Delta_native"])


def test_amplification_table_is_fdr_corrected_and_separates_real_from_null_series():
    idx = pd.date_range("2008-01-01", periods=204, freq="MS")
    masks, normal = _three_masks(idx)
    real = _series(seed=21, seasonal_amp=6.0, ar=0.3)
    real = real.copy()
    real[masks["Heatwave"]] += 2.0
    real[masks["Drought"]] += 1.0
    real[masks["Compound"]] += 7.0
    flat = _series(seed=22, seasonal_amp=6.0, ar=0.3)
    t = amplification_table(pd.DataFrame({"Real": real, "Flat": flat}), masks, normal, label="Region")
    t = t.set_index("Region")
    assert "q_value" in t.columns
    assert t.loc["Real", "Amplification_contrast"] > 2.0 and t.loc["Real", "p_value"] < 0.05
    assert t.loc["Flat", "CI_lo"] < 0 < t.loc["Flat", "CI_hi"]
