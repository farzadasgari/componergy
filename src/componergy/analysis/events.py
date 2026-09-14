from __future__ import annotations

import numpy as np
import xarray as xr


def compute_percentile_threshold(da: xr.DataArray, q: float) -> xr.DataArray:
    threshold = da.quantile(q, dim="time", skipna=True)
    return threshold.drop_vars("quantile", errors="ignore")


def classify_heatwave(sti: xr.DataArray, percentile: float = 0.90) -> xr.DataArray:
    threshold = compute_percentile_threshold(sti, percentile)
    return sti > threshold


def classify_drought(sapei: xr.DataArray, percentile: float = 0.10) -> xr.DataArray:
    threshold = compute_percentile_threshold(sapei, percentile)
    return sapei < threshold


def classify_compound(scdhi: xr.DataArray, percentile: float = 0.10) -> xr.DataArray:
    threshold = compute_percentile_threshold(scdhi, percentile)
    return scdhi < threshold
