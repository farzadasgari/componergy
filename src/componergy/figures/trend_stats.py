from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


def compute_area_weights(lat: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(lat))
    return weights / weights.mean()


def compute_statewide_mean(da: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    numerator = (da * weights).sum(dim=["lat", "lon"], skipna=True)
    valid_weight = xr.where(np.isfinite(da), weights, 0).sum(dim=["lat", "lon"])
    return numerator / valid_weight


def compute_footprint_fraction(is_extreme: xr.DataArray) -> xr.DataArray:
    return is_extreme.mean(dim=["lat", "lon"], skipna=True)


def annualize_monthly(da: xr.DataArray) -> xr.DataArray:
    return da.groupby("time.year").mean("time")


def compute_trend_per_decade(da_annual: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    years = da_annual["year"].astype(float)
    x = years - years.mean("year")
    n = da_annual.sizes["year"]
    x2sum = (x ** 2).sum("year")

    slope_per_year = (da_annual * x).sum("year") / x2sum
    slope_per_decade = slope_per_year * 10.0

    ybar = da_annual.mean("year")
    yhat = ybar + slope_per_year * x
    resid = da_annual - yhat
    df = n - 2
    sse = (resid ** 2).sum("year")
    sigma2 = sse / df
    se_slope = np.sqrt(sigma2 / x2sum)
    tstat = slope_per_year / se_slope

    pval = xr.apply_ufunc(
        lambda t: 2.0 * (1.0 - stats.t.cdf(np.abs(t), df=df)),
        tstat,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return slope_per_decade, pval
