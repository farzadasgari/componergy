import numpy as np
import pandas as pd
import xarray as xr

from componergy.indices.sti import add_sti, climatology_mean_std, climatology_sample_size


def _make_seasonal_dataset(n_years=10, seed=7):
    rng = np.random.default_rng(seed)
    lats = np.array([35.0, 38.0])
    lons = np.array([-120.0, -119.0])
    times = pd.date_range("2010-01-01", periods=12 * n_years, freq="MS")

    tavg = rng.normal(loc=15, scale=5, size=(len(times), len(lats), len(lons)))
    month_effect = 10 * np.sin(2 * np.pi * (times.month.values - 1) / 12)
    tavg = tavg + month_effect[:, None, None]

    return xr.Dataset(
        {"tavg": (("time", "lat", "lon"), tavg)},
        coords={"time": times, "lat": lats, "lon": lons},
    ), tavg, times


def test_sti_matches_manual_zscore_per_calendar_month():
    ds, tavg, times = _make_seasonal_dataset()
    result = add_sti(ds)

    max_abs_diff = 0.0
    for i_lat in range(tavg.shape[1]):
        for i_lon in range(tavg.shape[2]):
            series = tavg[:, i_lat, i_lon]
            months = times.month.values
            for m in range(1, 13):
                mask = months == m
                vals = series[mask]
                expected_mean = vals.mean()
                expected_std = vals.std(ddof=0)
                for idx in np.where(mask)[0]:
                    expected_z = (series[idx] - expected_mean) / expected_std
                    actual = float(result["sti"].isel(time=idx, lat=i_lat, lon=i_lon).values)
                    max_abs_diff = max(max_abs_diff, abs(actual - expected_z))

    assert max_abs_diff < 1e-10


def test_sti_is_deseasonalized():
    ds, _, _ = _make_seasonal_dataset()
    result = add_sti(ds)

    for m in [1, 6]:
        sti_month = result["sti"].isel(lat=0, lon=0).values[result["time.month"].values == m]
        assert abs(sti_month.mean()) < 1e-9
        assert abs(sti_month.std() - 1.0) < 1e-9


def test_sti_climatology_n_counts_years_correctly():
    ds, _, _ = _make_seasonal_dataset(n_years=10)
    result = add_sti(ds)
    n_vals = result["sti_climatology_n"].isel(lat=0, lon=0).values
    assert (n_vals == 10).all()


def test_sti_missing_month_propagates_nan_without_breaking_others():
    rng = np.random.default_rng(3)
    lats, lons = np.array([35.0]), np.array([-120.0])
    times = pd.date_range("2010-01-01", periods=60, freq="MS")
    tavg = rng.normal(loc=15, scale=5, size=(len(times), 1, 1))
    tavg[5, 0, 0] = np.nan  # one missing June

    ds = xr.Dataset({"tavg": (("time", "lat", "lon"), tavg)}, coords={"time": times, "lat": lats, "lon": lons})
    result = add_sti(ds)

    assert np.isnan(float(result["sti"].isel(time=5, lat=0, lon=0).values))
    n_june = float(result["sti_climatology_n"].sel(month=6).isel(lat=0, lon=0).values)
    assert n_june == 4  # 5 years of June, one missing

    june_stis = result["sti"].isel(lat=0, lon=0).values[result["time.month"].values == 6]
    assert np.isnan(june_stis[0])
    assert not np.any(np.isnan(june_stis[1:]))
