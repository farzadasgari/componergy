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
    one with the highest R² for a simple OLS fit of y on lagged x.

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
        (standard OLS slope t-test), reported because R² alone is
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
    """
    Benjamini-Hochberg FDR-adjusted p-values (q-values) for a collection of tests.

    Addresses the multiple-comparisons problem that arises whenever many
    p-values are examined together -- e.g. one best_lag_slope() p-value
    per (region, driver) pair, or one per (sector, driver) pair: with
    enough independent tests, some will look "significant" at p<0.05 by
    chance alone. Comparing each p-value to 0.05 in isolation does not
    control for this; comparing the returned q-values to a chosen FDR
    threshold (e.g. 0.05 or 0.10) does.

    NaN entries pass through as NaN and take no part in the ranking, so
    they never affect the q-values of the other entries.
    """
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


def _ols_ssr(X, y):
    """Residual sum of squares for an OLS fit of y on design matrix X."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return float(np.sum(resid ** 2))


def compute_incremental_r2(y, x_base, x_additional):
    """
    Nested-model comparison: does x_additional explain variance in y
    beyond what x_base already explains?

    Fits y ~ x_base (reduced model) and y ~ x_base + x_additional (full
    model), then runs a partial F-test on the reduction in residual sum
    of squares attributable to adding x_additional.

    Returns
    -------
    dict
        r2_reduced, r2_full, delta_r2, f_stat, p_value, n.
    """
    y = np.asarray(y, dtype=float)
    x_base = np.asarray(x_base, dtype=float)
    x_additional = np.asarray(x_additional, dtype=float)

    n = len(y)
    ones = np.ones(n)

    X_reduced = np.column_stack([ones, x_base])
    X_full = np.column_stack([ones, x_base, x_additional])

    sst = np.sum((y - y.mean()) ** 2)
    ssr_reduced = _ols_ssr(X_reduced, y)
    ssr_full = _ols_ssr(X_full, y)

    r2_reduced = 1.0 - ssr_reduced / sst if sst > 0 else np.nan
    r2_full = 1.0 - ssr_full / sst if sst > 0 else np.nan
    delta_r2 = r2_full - r2_reduced

    df_full = n - X_full.shape[1]
    df_diff = X_full.shape[1] - X_reduced.shape[1]

    if df_full > 0 and ssr_full > 0:
        f_stat = ((ssr_reduced - ssr_full) / df_diff) / (ssr_full / df_full)
        p_value = stats.f.sf(f_stat, df_diff, df_full)
    else:
        f_stat, p_value = np.nan, np.nan

    return {
        "r2_reduced": float(r2_reduced), "r2_full": float(r2_full), "delta_r2": float(delta_r2),
        "f_stat": float(f_stat) if np.isfinite(f_stat) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "n": int(n),
    }


def zscore(series: pd.Series) -> pd.Series:
    """Standardize a pandas Series to zero mean, unit variance. Returns all-zero if std is 0 or non-finite."""
    std = series.std(ddof=0)
    if not std or not np.isfinite(std):
        return series * 0.0
    return (series - series.mean()) / std


def deseasonalize_monthly(series: pd.Series) -> pd.Series:
    """Remove each calendar month's own climatological mean from a monthly series."""
    clim = series.groupby(series.index.month).mean()
    return series - series.index.month.map(clim)


def remove_linear_trend(series: pd.Series) -> pd.Series:
    """Fit and remove a linear trend over calendar time, returning the residuals."""
    valid = series.notna()
    idx = series.index[valid]
    y = series.values[valid].astype(float)

    years = idx.year.to_numpy(dtype=float) + (idx.month.to_numpy(dtype=float) - 0.5) / 12.0
    x = years

    xbar = x.mean()
    ybar = y.mean()
    sxx = np.sum((x - xbar) ** 2)
    if sxx == 0:
        residuals = pd.Series(np.nan, index=series.index)
        residuals.loc[idx] = y - ybar
        return residuals

    slope = np.sum((x - xbar) * (y - ybar)) / sxx
    intercept = ybar - slope * xbar
    fitted = intercept + slope * x

    residuals = pd.Series(np.nan, index=series.index)
    residuals.loc[idx] = y - fitted
    return residuals


def apply_fdr_to_grid(pvals: xr.DataArray) -> xr.DataArray:
    """Benjamini-Hochberg FDR correction across every cell of a gridded p-value field."""
    flat = pvals.values.flatten()
    q_flat = benjamini_hochberg_fdr(flat)
    q_grid = q_flat.reshape(pvals.shape)
    return xr.DataArray(q_grid, dims=pvals.dims, coords=pvals.coords)


def vectorized_welch_ttest_grid(da: xr.DataArray, mask_event: xr.DataArray, mask_normal: xr.DataArray) -> xr.DataArray:
    """Welch's t-test computed independently at every grid cell, vectorized (no per-cell Python loop)."""
    event_vals = da.where(mask_event)
    normal_vals = da.where(mask_normal)

    n_e = mask_event.sum(dim="time")
    n_n = mask_normal.sum(dim="time")
    mean_e = event_vals.mean(dim="time", skipna=True)
    mean_n = normal_vals.mean(dim="time", skipna=True)
    var_e = event_vals.var(dim="time", skipna=True, ddof=1)
    var_n = normal_vals.var(dim="time", skipna=True, ddof=1)

    se2 = var_e / n_e + var_n / n_n
    tstat = (mean_e - mean_n) / np.sqrt(se2)

    df = (se2 ** 2) / ((var_e / n_e) ** 2 / (n_e - 1) + (var_n / n_n) ** 2 / (n_n - 1))

    tstat_v = tstat.values
    df_v = df.values
    pval_v = np.full(tstat_v.shape, np.nan)
    valid = np.isfinite(tstat_v) & np.isfinite(df_v) & (df_v > 0)
    pval_v[valid] = 2 * stats.t.sf(np.abs(tstat_v[valid]), df_v[valid])

    return xr.DataArray(pval_v, dims=da.dims[1:], coords={d: da.coords[d] for d in da.dims[1:]})
