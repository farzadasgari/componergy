import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm

from componergy.indices.sapei import (
    add_sapei,
    water_balance,
    antecedent_water_balance,
    SAPEI_TIMESCALES_MONTHS,
)
from componergy.indices.loglogistic import fit_loglogistic_lmoments, loglogistic_cdf


def _make_test_dataset(n_years=30, seed=11, mask_one_cell=True):
    rng = np.random.default_rng(seed)
    lats = np.array([32.5, 35.0, 38.0])
    lons = np.array([-122.0, -120.0, -118.0])
    times = pd.date_range("2000-01-01", periods=12 * n_years, freq="MS")

    prcp = rng.gamma(shape=2.0, scale=20.0, size=(len(times), len(lats), len(lons)))
    pet = rng.uniform(20, 80, size=(len(times), len(lats), len(lons)))

    if mask_one_cell:
        prcp[:, 2, 2] = np.nan
        pet[:, 2, 2] = np.nan

    ds = xr.Dataset(
        {"prcp": (("time", "lat", "lon"), prcp), "pet": (("time", "lat", "lon"), pet)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    return ds, times


def test_water_balance_is_precip_minus_pet():
    ds, _ = _make_test_dataset()
    wb = water_balance(ds)
    expected = ds["prcp"].values - ds["pet"].values
    assert np.allclose(wb.values, expected, equal_nan=True)


def test_antecedent_water_balance_matches_manual_rolling_sum():
    ds, times = _make_test_dataset(n_years=5, mask_one_cell=False)
    wb = water_balance(ds)
    wsd3 = antecedent_water_balance(wb, 3).isel(lat=0, lon=0).values
    manual = wb.isel(lat=0, lon=0).values

    assert np.isnan(wsd3[0]) and np.isnan(wsd3[1])
    for i in range(2, len(manual)):
        expected = manual[i - 2] + manual[i - 1] + manual[i]
        assert abs(wsd3[i] - expected) < 1e-9
