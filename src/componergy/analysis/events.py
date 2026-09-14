from __future__ import annotations

import numpy as np
import xarray as xr


def compute_percentile_threshold(da: xr.DataArray, q: float) -> xr.DataArray:
    threshold = da.quantile(q, dim="time", skipna=True)
    return threshold.drop_vars("quantile", errors="ignore")
