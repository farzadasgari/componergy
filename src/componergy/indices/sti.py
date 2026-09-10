from __future__ import annotations

import xarray as xr


def climatology_mean_std(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    grouped = da.groupby("time.month")
    mean = grouped.mean("time")
    std = grouped.std("time")
    return mean, std


def climatology_sample_size(da: xr.DataArray) -> xr.DataArray:
    return da.groupby("time.month").count("time")