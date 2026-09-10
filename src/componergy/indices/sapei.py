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