import numpy as np
import pandas as pd
import xarray as xr

from componergy.figures.trend_stats import (
    compute_area_weights,
    compute_statewide_mean,
    compute_footprint_fraction,
    annualize_monthly,
    compute_trend_per_decade,
    compute_linear_trend_with_ci,
)


def test_area_weights_match_manual_cosine_calc():
    lats = xr.DataArray([0.0, 30.0, 60.0, 90.0], dims="lat")
    w = compute_area_weights(lats)
    expected = np.cos(np.deg2rad(lats.values))
    expected = expected / expected.mean()
    assert np.allclose(w.values, expected)


def test_statewide_mean_excludes_nan_cell_from_numerator_and_denominator():
    lats = xr.DataArray([0.0, 60.0], dims="lat")
    lons = xr.DataArray([-120.0, -119.0], dims="lon")
    times = pd.date_range("2000-01-01", periods=2, freq="MS")
    data = np.array([
        [[10.0, 20.0], [30.0, 40.0]],
        [[11.0, 21.0], [np.nan, 41.0]],
    ])
    da = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": times, "lat": lats, "lon": lons})
    weights = compute_area_weights(lats).broadcast_like(da.isel(time=0))

    result = compute_statewide_mean(da, weights)

    w_vals = compute_area_weights(lats).values
    w_grid = np.array([[w_vals[0], w_vals[0]], [w_vals[1], w_vals[1]]])
    month2 = data[1]
    valid = np.isfinite(month2)
    expected_month2 = np.sum(month2[valid] * w_grid[valid]) / np.sum(w_grid[valid])

    assert abs(float(result.isel(time=1).values) - expected_month2) < 1e-9


def test_footprint_fraction_matches_manual_calc():
    lats = xr.DataArray([0.0, 60.0], dims="lat")
    lons = xr.DataArray([-120.0, -119.0], dims="lon")
    bool_da = xr.DataArray(
        np.array([[True, False], [True, True]]),
        dims=("lat", "lon"), coords={"lat": lats, "lon": lons},
    )
    frac = compute_footprint_fraction(bool_da)
    assert abs(float(frac.values) - 0.75) < 1e-9


def _make_known_trend_grid(true_slope_per_year, n_years=30, noise_std=0.0, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("1990-01-01", periods=12 * n_years, freq="MS")
    years_frac = times.year.values + (times.month.values - 0.5) / 12.0
    values = true_slope_per_year * (years_frac - years_frac.mean())
    if noise_std:
        values = values + rng.normal(0, noise_std, len(times))
    da = xr.DataArray(
        np.broadcast_to(values[:, None, None], (len(times), 2, 2)).copy(),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [35.0, 38.0], "lon": [-120.0, -119.0]},
    )
    return da, times, values


def test_trend_per_decade_recovers_exact_noiseless_slope():
    da, _, _ = _make_known_trend_grid(true_slope_per_year=0.5, noise_std=0.0)
    annual = annualize_monthly(da)
    slope_decade, _ = compute_trend_per_decade(annual)
    assert abs(float(slope_decade.isel(lat=0, lon=0).values) - 5.0) < 1e-6

