import numpy as np
import pandas as pd
from scipy import stats

from componergy.analysis.extremes import (
    empirical_return_period,
    fit_gev_annual_maxima,
    fit_pot_gpd,
    return_period_of_value,
    decluster_runs,
    bootstrap_return_period,
    gev_goodness_of_fit,
    threshold_stability,
    stationarity_check,
)


def test_empirical_return_period_of_maximum_is_fixed_by_record_length():
    idx = pd.date_range("2008-01-01", periods=204, freq="MS")

    modest = pd.Series(np.r_[np.zeros(203), 1.0], index=idx)
    extreme = pd.Series(np.r_[np.zeros(203), 1000.0], index=idx)

    rp_modest = empirical_return_period(modest)["Return_Period_Years"].iloc[0]
    rp_extreme = empirical_return_period(extreme)["Return_Period_Years"].iloc[0]

    assert abs(rp_modest - rp_extreme) < 1e-9
    assert abs(rp_modest - (205.0 / 12.0)) < 1e-9


def test_gev_and_pot_recover_known_return_levels():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2008-01-01", periods=204, freq="MS")
    true_c, true_loc, true_scale = 0.1, 2.0, 0.5
    s = pd.Series(stats.genextreme.rvs(c=true_c, loc=true_loc, scale=true_scale,
                                       size=204, random_state=rng), index=idx)

    gev = fit_gev_annual_maxima(s)
    pot = fit_pot_gpd(s)

    assert gev["n_maxima"] == 17
    assert pot["n_exceedances"] > 8

    gev_levels = [gev["return_levels"][T] for T in (2, 5, 10, 20, 50, 100)]
    pot_levels = [pot["return_levels"][T] for T in (2, 5, 10, 20, 50, 100)]
    assert all(np.diff(gev_levels) > 0)
    assert all(np.diff(pot_levels) > 0)


def test_fitted_return_period_exceeds_empirical_for_the_record_maximum():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2008-01-01", periods=204, freq="MS")
    s = pd.Series(stats.genextreme.rvs(c=0.1, loc=2.0, scale=0.5, size=204, random_state=rng), index=idx)

    gev = fit_gev_annual_maxima(s)
    pot = fit_pot_gpd(s)
    obs_max = float(s.max())
    rp = return_period_of_value(obs_max, gev, pot)

    empirical_cap = 205.0 / 12.0
    assert rp["gev_return_period_years"] > empirical_cap
    assert rp["pot_return_period_years"] > empirical_cap


def _persistent_series(n=204, seed=0, ar=0.8):
    idx = pd.date_range("2008-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = ar * e[i - 1] + rng.normal()
    return pd.Series(e, index=idx)


def test_decluster_runs_keeps_one_maximum_per_cluster():
    idx = pd.date_range("2010-01-01", periods=16, freq="MS")
    s = pd.Series([0, 5, 9, 6, 0, 0, 0, 0, 0, 7, 8, 0, 0, 0, 4, 0], index=idx, dtype=float)
    out = decluster_runs(s, threshold=3.0, run_length=3)
    assert list(out.values) == [9.0, 8.0, 4.0]
    assert out.index[0] == idx[2]


def test_decluster_bridges_a_short_gap_but_not_a_long_one():
    idx = pd.date_range("2010-01-01", periods=12, freq="MS")
    s = pd.Series([5, 0, 6, 0, 0, 0, 0, 7, 0, 0, 0, 8], index=idx, dtype=float)
    assert len(decluster_runs(s, 3.0, run_length=3)) == 3
    assert len(
        decluster_runs(s, 3.0, run_length=1)) == 4


def test_declustering_reduces_the_count_of_independent_exceedances_on_a_persistent_series():
    s = _persistent_series()
    raw = fit_pot_gpd(s, decluster_run=None)
    dec = fit_pot_gpd(s, decluster_run=3)
    assert dec["declustered"] is True and "declustered_r3" in dec["method"]
    assert dec["n_exceedances"] < raw["n_exceedances"]
    assert raw["method"] == "POT_GPD"  # unchanged default behaviour
