"""Standardized Antecedent Precipitation Evapotranspiration Index (SAPEI), monthly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm import tqdm

from componergy.indices.loglogistic import fit_loglogistic_lmoments, loglogistic_cdf

SAPEI_TIMESCALES_MONTHS = (3, 6, 9, 12)
MIN_SAMPLES_PER_MONTH = 10


def water_balance(monthly_ds: xr.Dataset) -> xr.DataArray:
    """Monthly water balance (precipitation - PET), vectorized grid-wide."""
    return monthly_ds["prcp"] - monthly_ds["pet"]


def antecedent_water_balance(wb: xr.DataArray, n_months: int) -> xr.DataArray:
    """Rolling n-month sum of water balance, vectorized grid-wide."""
    return wb.rolling(time=n_months, min_periods=n_months).sum()


def compute_sapei_for_cell(wsd_by_timescale: dict, months: np.ndarray) -> dict:
    """
    Fit log-logistic per calendar month and standardize, for ONE grid cell.

    Handles all requested timescales for that cell in a single call, to
    keep per-task overhead low when parallelized across cells.

    Parameters
    ----------
    wsd_by_timescale : dict[int, np.ndarray]
        {n_months: antecedent water balance series} for one cell.
    months : np.ndarray
        Calendar month (1-12) for each timestep, same length as each series.

    Returns
    -------
    dict[int, np.ndarray]
        {n_months: sapei series}, NaN wherever a calendar-month group had
        fewer than MIN_SAMPLES_PER_MONTH valid values.
    """
    result = {}
    for n_months, wsd in wsd_by_timescale.items():
        sapei = np.full_like(wsd, np.nan)
        for m in range(1, 13):
            mask = months == m
            group_vals = wsd[mask]
            valid = ~np.isnan(group_vals)
            if valid.sum() < MIN_SAMPLES_PER_MONTH:
                continue
            alpha, beta, gamma_param = fit_loglogistic_lmoments(group_vals[valid])
            q = np.clip(loglogistic_cdf(group_vals[valid], alpha, beta, gamma_param), 1e-10, 1 - 1e-10)
            month_indices = np.where(mask)[0]
            sapei[month_indices[valid]] = norm.ppf(q)
        result[n_months] = sapei
    return result


def add_sapei(monthly_ds: xr.Dataset, n_jobs: int = -1) -> xr.Dataset:
    """
    Add sapei_3m/6m/9m/12m to a monthly climate dataset.

    Cells that are NaN across the entire record (e.g. ocean/masked cells
    left over from the California clip) are skipped rather than fit.

    Parameters
    ----------
    monthly_ds : xr.Dataset
        Must contain 'prcp' and 'pet' on a monthly time dimension.
    n_jobs : int

    Returns
    -------
    xr.Dataset
        `monthly_ds` with sapei_3m, sapei_6m, sapei_9m, sapei_12m added.
    """
    wb = water_balance(monthly_ds)
    wsd_grids = {n: antecedent_water_balance(wb, n) for n in SAPEI_TIMESCALES_MONTHS}

    times = pd.DatetimeIndex(monthly_ds["time"].values)
    months = times.month.values
    lats = monthly_ds["lat"].values
    lons = monthly_ds["lon"].values

    valid_cell = ~np.isnan(wb.values).all(axis=0)
    cell_indices = [(i, j) for i in range(len(lats)) for j in range(len(lons)) if valid_cell[i, j]]

    cell_inputs = [
        {n: wsd_grids[n].values[:, i, j] for n in SAPEI_TIMESCALES_MONTHS}
        for i, j in cell_indices
    ]

    cell_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_sapei_for_cell)(wsd_by_timescale, months)
        for wsd_by_timescale in tqdm(cell_inputs, desc="fitting SAPEI per cell", unit="cell")
    )

    sapei_arrays = {n: np.full((len(times), len(lats), len(lons)), np.nan) for n in SAPEI_TIMESCALES_MONTHS}
    for (i_lat, i_lon), cell_result in zip(cell_indices, cell_results):
        for n in SAPEI_TIMESCALES_MONTHS:
            sapei_arrays[n][:, i_lat, i_lon] = cell_result[n]

    out = monthly_ds.copy()
    for n in SAPEI_TIMESCALES_MONTHS:
        out[f"sapei_{n}m"] = (("time", "lat", "lon"), sapei_arrays[n])
        out[f"sapei_{n}m"].attrs[
            "long_name"] = f"Standardized Antecedent Precipitation Evapotranspiration Index ({n}-month)"
    return out
