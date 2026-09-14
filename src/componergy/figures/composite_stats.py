"""Composite event-comparison and lag-scan regression statistics for time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


def series_to_data_array(series: pd.Series) -> xr.DataArray:
    return xr.DataArray(series.values, dims="time", coords={"time": series.index})
