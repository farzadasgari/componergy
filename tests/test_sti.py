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
