from __future__ import annotations

import xarray as xr


def climatology_mean_std(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    grouped = da.groupby("time.month")
    mean = grouped.mean("time")
    std = grouped.std("time")
    return mean, std


def climatology_sample_size(da: xr.DataArray) -> xr.DataArray:
    return da.groupby("time.month").count("time")


def add_sti(monthly_ds: xr.Dataset, temp_var: str = "tavg") -> xr.Dataset:
    da = monthly_ds[temp_var]
    mean, std = climatology_mean_std(da)
    n = climatology_sample_size(da)

    anomaly = da.groupby("time.month") - mean
    sti = anomaly.groupby("time.month") / std

    out = monthly_ds.copy()
    out["sti"] = sti
    out["sti_climatology_mean"] = mean
    out["sti_climatology_std"] = std
    out["sti_climatology_n"] = n
    return out
