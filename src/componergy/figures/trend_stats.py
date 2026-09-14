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
