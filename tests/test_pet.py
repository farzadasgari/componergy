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

# Fresno-like monthly climate normals (degC), Jan..Dec, Central Valley ~36.8 N.
CENTRAL_VALLEY_LAT = 36.8
CENTRAL_VALLEY_TMAX = [13.6, 17.2, 20.6, 24.2, 28.9, 33.6, 36.8, 36.1, 32.8, 26.8, 18.9, 13.2]
CENTRAL_VALLEY_TMIN = [2.9, 4.3, 6.5, 8.6, 12.3, 16.2, 19.4, 18.7, 15.8, 11.1, 5.6, 2.7]

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


def test_hargreaves_pet_monthly_matches_fao56_formula_exactly():
    lat, month, tavg, tmax, tmin, days = 36.8, 7, 28.0, 36.0, 19.0, 31
    ra_mj = _extraterrestrial_radiation(lat, 30.44 * (month - 1) + 15)
    expected = 0.0023 * (0.408 * ra_mj) * (tavg + 17.8) * np.sqrt(tmax - tmin) * days
    assert hargreaves_pet_monthly(lat, month, tavg, tmax, tmin, days) == pytest.approx(expected, rel=1e-12)


def test_annual_pet_is_physically_plausible_for_central_valley():
    annual = 0.0
    for month in range(1, 13):
        tmax = CENTRAL_VALLEY_TMAX[month - 1]
        tmin = CENTRAL_VALLEY_TMIN[month - 1]
        tavg = (tmax + tmin) / 2
        days = calendar.monthrange(2021, month)[1]
        annual += hargreaves_pet_monthly(CENTRAL_VALLEY_LAT, month, tavg, tmax, tmin, days)
    assert 1000 < annual < 2000, f"annual PET {annual:.0f} mm outside 1000-2000 mm"


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
