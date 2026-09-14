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
