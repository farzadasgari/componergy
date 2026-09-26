"""Area-weighted spatial aggregation and linear trend statistics for gridded time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


def compute_area_weights(lat: xr.DataArray) -> xr.DataArray:
    """Latitude weights proportional to cos(latitude), normalized to a mean of 1."""
    weights = np.cos(np.deg2rad(lat))
    return weights / weights.mean()


def compute_statewide_mean(da: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    """
    Area-weighted mean over lat/lon, preserving the time dimension.

    NaN cells are excluded from both the weighted sum and the weight
    total, so they don't bias the result toward zero.
    """
    numerator = (da * weights).sum(dim=["lat", "lon"], skipna=True)
    valid_weight = xr.where(np.isfinite(da), weights, 0).sum(dim=["lat", "lon"])
    return numerator / valid_weight


def compute_footprint_fraction(is_extreme: xr.DataArray) -> xr.DataArray:
    """Fraction of grid cells where `is_extreme` is True, per timestep."""
    return is_extreme.mean(dim=["lat", "lon"], skipna=True)


def annualize_monthly(da: xr.DataArray) -> xr.DataArray:
    """Mean of monthly values within each calendar year."""
    return da.groupby("time.year").mean("time")


def compute_trend_per_decade(da_annual: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Ordinary least-squares linear trend per decade, per grid cell, with a two-tailed p-value."""
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


def compute_linear_trend_with_ci(series: pd.Series, confidence: float = 0.95) -> tuple[pd.Series, pd.Series]:
    """Fit an OLS line to a 1-D time series and return the fitted values and confidence-interval half-width."""
    valid = series.notna()
    idx = series.index[valid]
    y = series.values[valid].astype(float)

    years = idx.year.to_numpy(dtype=float) + (idx.month.to_numpy(dtype=float) - 0.5) / 12.0
    x = years

    n = len(x)
    xbar = x.mean()
    ybar = y.mean()
    sxx = np.sum((x - xbar) ** 2)

    slope = np.sum((x - xbar) * (y - ybar)) / sxx
    intercept = ybar - slope * xbar
    yhat = intercept + slope * x

    resid = y - yhat
    df = n - 2
    sse = np.sum(resid ** 2)
    sigma2 = sse / df
    se_yhat = np.sqrt(sigma2 * (1 / n + (x - xbar) ** 2 / sxx))
    tcrit = stats.t.ppf(1 - (1 - confidence) / 2, df=df)
    ci = tcrit * se_yhat

    fitted = pd.Series(yhat, index=idx)
    ci_band = pd.Series(ci, index=idx)
    return fitted, ci_band
