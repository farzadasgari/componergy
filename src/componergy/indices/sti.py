"""
Standardized Temperature Index (STI), monthly.

STI is a per-calendar-month z-score of monthly-mean temperature,
pooling all years -- mathematically equivalent to fitting a normal distribution to each
calendar month's temperature and taking the standard-normal quantile of
each year's value (HESS 2021, Eqs. 2-3), since for a Gaussian,
Phi^-1(Phi((x-mu)/sigma)) = (x-mu)/sigma.

Positive STI = warmer than that calendar month's long-term normal.
"""

from __future__ import annotations

import xarray as xr


def climatology_mean_std(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Per-calendar-month mean and standard deviation, pooling all years.

    Returns two DataArrays indexed by 'month' (1-12) plus the original
    non-time dims (e.g. lat, lon), not by 'time' -- one climatology value
    per calendar month, not one per timestep.
    """
    grouped = da.groupby("time.month")
    mean = grouped.mean("time")
    std = grouped.std("time")
    return mean, std


def climatology_sample_size(da: xr.DataArray) -> xr.DataArray:
    """
    Number of valid (non-NaN) years contributing to each calendar month's climatology.

    A useful diagnostic for flagging thin climatologies (e.g. near the
    start of the record, or where source data has gaps).
    """
    return da.groupby("time.month").count("time")


def add_sti(monthly_ds: xr.Dataset, temp_var: str = "tavg") -> xr.Dataset:
    """
    Add sti to a monthly climate dataset.

    Also adds sti_climatology_mean/std/n as diagnostic variables (indexed
    by calendar month, not time) so the fitted climatology used to
    standardize each value stays inspectable.

    Parameters
    ----------
    monthly_ds : xr.Dataset
        Must contain `temp_var` on a monthly time dimension, as produced
        by preprocessing.aggregate_monthly.
    temp_var : str
        Name of the temperature variable to standardize (default "tavg").

    Returns
    -------
    xr.Dataset
        `monthly_ds` with 'sti', 'sti_climatology_mean', 'sti_climatology_std',
        and 'sti_climatology_n' added.
    """
    da = monthly_ds[temp_var]
    mean, std = climatology_mean_std(da)
    n = climatology_sample_size(da)

    anomaly = da.groupby("time.month") - mean
    sti = anomaly.groupby("time.month") / std

    out = monthly_ds.copy()
    out["sti"] = sti
    out["sti"].attrs["long_name"] = "Standardized Temperature Index"
    out["sti_climatology_mean"] = mean
    out["sti_climatology_std"] = std
    out["sti_climatology_n"] = n
    return out
