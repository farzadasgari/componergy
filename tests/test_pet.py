import calendar

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from componergy.indices.pet import (
    _extraterrestrial_radiation,
    add_pet,
    hargreaves_pet_monthly,
)

CALIFORNIA_LATS = [32.5, 35.0, 38.0, 41.5]


def test_ra_never_negative_across_california_latitudes_and_months():
    for lat in CALIFORNIA_LATS:
        for month in range(1, 13):
            doy = 30.44 * (month - 1) + 15
            ra = _extraterrestrial_radiation(lat, doy)
            assert ra >= 0, f"Ra negative at lat={lat}, month={month}: {ra}"


def test_ra_seasonal_pattern_winter_lower_than_summer():
    ra_jan = _extraterrestrial_radiation(35.0, 30.44 * 0 + 15)
    ra_jul = _extraterrestrial_radiation(35.0, 30.44 * 6 + 15)
    assert ra_jan < ra_jul


def test_hargreaves_pet_monthly_nan_propagates():
    assert np.isnan(hargreaves_pet_monthly(35.0, 1, np.nan, 20.0, 10.0, 31))


def test_hargreaves_pet_monthly_zero_trange_is_zero():
    pet = hargreaves_pet_monthly(35.0, 1, 10.0, 10.0, 10.0, 31)
    assert pet == pytest.approx(0.0)


def test_hargreaves_pet_monthly_january_is_not_zero():
    pet = hargreaves_pet_monthly(35.0, 1, 10.0, 15.0, 5.0, 31)
    assert pet > 0


def test_add_pet_matches_scalar_reference_exactly():
    rng = np.random.default_rng(0)
    lats = np.array(CALIFORNIA_LATS)
    lons = np.array([-124.0, -120.0, -116.0])
    times = pd.date_range("2020-01-01", periods=12, freq="MS")

    tavg = rng.uniform(0, 30, size=(len(times), len(lats), len(lons)))
    tmax = tavg + rng.uniform(2, 15, size=tavg.shape)
    tmin = tavg - rng.uniform(2, 15, size=tavg.shape)

    ds = xr.Dataset(
        {
            "tavg": (("time", "lat", "lon"), tavg),
            "tmax": (("time", "lat", "lon"), tmax),
            "tmin": (("time", "lat", "lon"), tmin),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    result = add_pet(ds)

    for i_t, t in enumerate(times):
        days_in_month = calendar.monthrange(t.year, t.month)[1]
        for i_lat, lat in enumerate(lats):
            for i_lon in range(len(lons)):
                expected = hargreaves_pet_monthly(
                    lat, t.month,
                    float(tavg[i_t, i_lat, i_lon]),
                    float(tmax[i_t, i_lat, i_lon]),
                    float(tmin[i_t, i_lat, i_lon]),
                    days_in_month,
                )
                actual = float(result["pet"].isel(time=i_t, lat=i_lat, lon=i_lon).values)
                assert actual == pytest.approx(expected, abs=1e-8)
