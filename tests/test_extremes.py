import numpy as np
import pandas as pd
from scipy import stats

from componergy.analysis.extremes import (
    empirical_return_period, fit_gev_annual_maxima, fit_pot_gpd, return_period_of_value,
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
