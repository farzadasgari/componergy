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
