import numpy as np
import pandas as pd
import xarray as xr

from componergy.figures.hovmoller import reduce_for_hovmoller, symmetric_levels


def _ds(n_months=120, n_lat=6, n_lon=5):
    times = pd.date_range("2000-01-01", periods=n_months, freq="MS")
    lats = np.linspace(33, 41, n_lat)
    lons = np.linspace(-123, -115, n_lon)
    rng = np.random.default_rng(0)
    return xr.DataArray(rng.normal(0, 1, (n_months, n_lat, n_lon)),
                        dims=("time", "lat", "lon"),
                        coords={"time": times, "lat": lats, "lon": lons})


def test_reduce_annual_collapses_months_and_keeps_spatial_axis():
    da = _ds(n_months=120, n_lat=6)
    out = reduce_for_hovmoller(da, mode="annual", space="lat")
    assert out.dims == ("year", "lat")
    assert out.sizes["year"] == 10
    assert out.sizes["lat"] == 6


def test_reduce_monthly_keeps_every_month():
    da = _ds(n_months=120, n_lon=5)
    out = reduce_for_hovmoller(da, mode="monthly", space="lon")
    assert out.dims == ("time", "lon")
    assert out.sizes["time"] == 120
    assert out.sizes["lon"] == 5


def test_symmetric_levels_are_centred_on_zero():
    da = _ds()
    levels = symmetric_levels(da)
    assert abs(levels[0] + levels[-1]) < 1e-9
    assert np.all(np.diff(levels) > 0)


def test_symmetric_levels_survive_all_nan_input():
    da = _ds() * np.nan
    levels = symmetric_levels(da)
    assert np.all(np.isfinite(levels))
    assert abs(levels[0] + levels[-1]) < 1e-9
