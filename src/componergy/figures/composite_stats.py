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


def compute_composite_table(signal_df: pd.DataFrame, event_masks: dict, normal_mask: pd.Series) -> pd.DataFrame:
    rows = []
    for event_name, event_mask in event_masks.items():
        event_mask = event_mask.reindex(signal_df.index).fillna(False)
        for col in signal_df.columns:
            event_vals = signal_df.loc[event_mask, col]
            normal_vals = signal_df.loc[normal_mask, col]
            diff = float(event_vals.mean() - normal_vals.mean())
            tstat, pval = welch_ttest(event_vals, normal_vals)
            rows.append({
                "Event": event_name,
                "Source": col,
                "N_event": int(event_mask.sum()),
                "N_normal": int(normal_mask.sum()),
                "Mean_event": float(event_vals.mean()),
                "Mean_normal": float(normal_vals.mean()),
                "Difference": diff,
                "tstat": tstat,
                "p_value": pval,
            })
    return pd.DataFrame(rows)
