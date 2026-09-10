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
