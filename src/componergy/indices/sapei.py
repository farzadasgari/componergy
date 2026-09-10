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
    return wb.rolling(time=n_months, min_periods=n_months).sum()


def compute_sapei_for_cell(wsd_by_timescale: dict, months: np.ndarray) -> dict:
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
    wb = water_balance(monthly_ds)
    wsd_grids = {n: antecedent_water_balance(wb, n) for n in SAPEI_TIMESCALES_MONTHS}

    times = pd.DatetimeIndex(monthly_ds["time"].values)
    months = times.month.values
    lats = monthly_ds["lat"].values
    lons = monthly_ds["lon"].values

    valid_cell = ~np.isnan(wb.values).all(axis=0)
    cell_indices = [(i, j) for i in range(len(lats)) for j in range(len(lons)) if valid_cell[i, j]]

    def process_cell(i_lat, i_lon):
        wsd_by_timescale = {n: wsd_grids[n].values[:, i_lat, i_lon] for n in SAPEI_TIMESCALES_MONTHS}
        return i_lat, i_lon, compute_sapei_for_cell(wsd_by_timescale, months)

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cell)(i, j)
        for i, j in tqdm(cell_indices, desc="fitting SAPEI per cell", unit="cell")
    )

    sapei_arrays = {n: np.full((len(times), len(lats), len(lons)), np.nan) for n in SAPEI_TIMESCALES_MONTHS}
    for i_lat, i_lon, cell_result in results:
        for n in SAPEI_TIMESCALES_MONTHS:
            sapei_arrays[n][:, i_lat, i_lon] = cell_result[n]

    out = monthly_ds.copy()
    for n in SAPEI_TIMESCALES_MONTHS:
        out[f"sapei_{n}m"] = (("time", "lat", "lon"), sapei_arrays[n])
        out[f"sapei_{n}m"].attrs[
            "long_name"] = f"Standardized Antecedent Precipitation Evapotranspiration Index ({n}-month)"
    return out
