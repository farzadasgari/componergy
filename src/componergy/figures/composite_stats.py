"""Composite event-comparison and lag-scan regression statistics for time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


def series_to_data_array(series: pd.Series) -> xr.DataArray:
    """
    Wrap a pandas Series as a 1-D xr.DataArray with dim 'time'.

    Lets the existing xarray-based event classification functions
    (analysis.events) be reused directly on a statewide-aggregated
    (non-gridded) signal, without a separate pandas-specific
    reimplementation of the same threshold logic.
    """
    return xr.DataArray(series.values, dims="time", coords={"time": series.index})


def welch_ttest(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """
    Welch's t-test (unequal variance) between two samples, dropping NaNs first.

    Returns (nan, nan) if either sample has fewer than 5 valid values.
    """
    a = a.dropna()
    b = b.dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    tstat, pval = stats.ttest_ind(a, b, equal_var=False)
    return float(tstat), float(pval)


def compute_composite_table(signal_df: pd.DataFrame, event_masks: dict, normal_mask: pd.Series) -> pd.DataFrame:
    """
    For every (event, column) pair, the mean-difference anomaly and a
    Welch's t-test against the normal-period sample.

    Parameters
    ----------
    signal_df : pd.DataFrame
        One column per series to test (e.g. one per generation source).
    event_masks : dict[str, pd.Series]
        {event_name: boolean mask}, each aligned or reindexable to signal_df's index.
    normal_mask : pd.Series
        Boolean mask defining the reference/baseline period.

    Returns
    -------
    pd.DataFrame
        One row per (event, column): N_event, N_normal, Mean_event,
        Mean_normal, Difference, tstat, p_value.
    """
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


def best_lag_slope(x: pd.Series, y: pd.Series, lags=range(-6, 7), min_n=60):
    """
    Scan a range of lags (in the series' own time step) and return the
    one with the highest R^2 for a simple OLS fit of y on lagged x.

    A positive lag means x is shifted to lead y by that many steps.

    Parameters
    ----------
    x, y : pd.Series
        Same-frequency series (need not already be aligned/same length).
    lags : iterable[int]
        Candidate lags to scan.
    min_n : int
        Minimum number of valid paired observations required to
        consider a given lag.

    Returns
    -------
    dict or None
        {"lag", "slope", "r2", "n", "p_value"} for the best lag, or None
        if no lag met min_n. p_value tests H0: slope == 0 at that lag
        (standard OLS slope t-test) -- reported because R² alone is
        descriptive; whether the best-fit slope is distinguishable from
        no relationship is a separate question a bare R² doesn't answer.
        When comparing this p-value across many regions/drivers/sectors,
        apply benjamini_hochberg_fdr() to the collected p-values rather
        than reading each one in isolation.
    """
    best = None
    for lag in lags:
        if lag < 0:
            x_lagged = x.iloc[-lag:].values
            y_lagged = y.iloc[:len(x_lagged)].values
        elif lag > 0:
            x_lagged = x.iloc[:-lag].values
            y_lagged = y.iloc[lag:].values
        else:
            x_lagged = x.values
            y_lagged = y.values

        mask = np.isfinite(x_lagged) & np.isfinite(y_lagged)
        n = int(mask.sum())
        if n < min_n:
            continue

        xv, yv = x_lagged[mask], y_lagged[mask]
        A = np.vstack([xv, np.ones(n)]).T
        slope, intercept = np.linalg.lstsq(A, yv, rcond=None)[0]
        yhat = slope * xv + intercept
        ssr = np.sum((yv - yhat) ** 2)
        sst = np.sum((yv - np.mean(yv)) ** 2)
        r2 = 1.0 - ssr / sst if sst > 0 else np.nan

        xbar = xv.mean()
        sxx = np.sum((xv - xbar) ** 2)
        mse = ssr / (n - 2) if n > 2 else np.nan
        se_slope = np.sqrt(mse / sxx) if sxx > 0 and np.isfinite(mse) else np.nan
        tstat = slope / se_slope if se_slope and np.isfinite(se_slope) else np.nan
        p_value = 2 * stats.t.sf(np.abs(tstat), df=n - 2) if np.isfinite(tstat) else np.nan

        candidate = {"lag": int(lag), "slope": float(slope), "r2": float(r2), "n": n,
                     "p_value": float(p_value) if np.isfinite(p_value) else np.nan}
        if best is None or np.nan_to_num(candidate["r2"], nan=-np.inf) > np.nan_to_num(best["r2"], nan=-np.inf):
            best = candidate
    return best


def benjamini_hochberg_fdr(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    q_values = np.full(n, np.nan)

    valid_mask = np.isfinite(p_values)
    valid_p = p_values[valid_mask]
    n_valid = len(valid_p)
    if n_valid == 0:
        return q_values

    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    ranks = np.arange(1, n_valid + 1)

    raw_q = ranked_p * n_valid / ranks
    adjusted_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    adjusted_q = np.clip(adjusted_q, 0, 1)

    valid_q = np.empty(n_valid)
    valid_q[order] = adjusted_q
    q_values[valid_mask] = valid_q
    return q_values
