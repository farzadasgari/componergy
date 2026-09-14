"""Composite event-comparison and lag-scan regression statistics for time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


def series_to_data_array(series: pd.Series) -> xr.DataArray:
    return xr.DataArray(series.values, dims="time", coords={"time": series.index})


def welch_ttest(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    a = a.dropna()
    b = b.dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    tstat, pval = stats.ttest_ind(a, b, equal_var=False)
    return float(tstat), float(pval)
