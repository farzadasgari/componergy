import numpy as np
import pandas as pd
import pytest

from componergy.analysis.regression import (
    design_matrix,
    newey_west_cov,
    ols_hac,
    event_effects,
    amplification_contrast,
    amplification_table,
    adjusted_composite_table,
    normal_baseline_anomalies,
    baseline_adjusted_z,
    randomization_inference,
    shift_range,
    shift_null_coefficients,
    year_fraction,
)


def _series(n=204, seed=0, seasonal_amp=10.0, trend_per_decade=0.0, ar=0.5, noise=1.0, start="2008-01-01"):
    idx = pd.date_range(start, periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    season = seasonal_amp * np.sin(2 * np.pi * (idx.month.values - 4) / 12)
    yr = year_fraction(idx)
    trend = trend_per_decade * (yr - yr.mean()) / 10.0
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = ar * e[i - 1] + rng.normal(0, noise)
    return pd.Series(season + trend + e, index=idx)


def _mask(idx, months_by_year):
    keys = {(y, m) for y, ms in months_by_year.items() for m in ms}
    return pd.Series([(t.year, t.month) in keys for t in idx], index=idx)


def _three_masks(idx):
    m = {
        "Heatwave": _mask(idx, {2009: [6, 7, 8], 2012: [7, 8], 2015: [6, 7, 8], 2018: [7, 8]}),
        "Drought": _mask(idx, {2010: [5, 6, 7], 2013: [6, 7], 2016: [5, 6, 7], 2019: [6, 7]}),
        "Compound": _mask(idx, {2011: [7, 8, 9], 2014: [7, 8], 2017: [7, 8, 9], 2021: [6, 7, 8]}),
    }
    return m, ~(m["Heatwave"] | m["Drought"] | m["Compound"])
