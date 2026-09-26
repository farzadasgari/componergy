"""
Extreme-value analysis of the systemic risk series: empirical plotting
positions plus two fitted extreme-value models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MONTHS_PER_YEAR = 12


def empirical_return_period(series: pd.Series) -> pd.DataFrame:
    """Weibull plotting position (n+1)/rank, converted to years."""
    s = series.dropna().sort_values(ascending=False)
    n = len(s)
    ranks = np.arange(1, n + 1)
    return pd.DataFrame({
        "Date": s.index,
        "Value": s.values,
        "Return_Period_Years": ((n + 1.0) / ranks) / MONTHS_PER_YEAR,
    })


def fit_gev_annual_maxima(series: pd.Series, return_periods_years=(2, 5, 10, 20, 50, 100)):
    """Fit a GEV to annual maxima and return fitted return levels."""
    s = series.dropna()
    if len(s) < MONTHS_PER_YEAR * 5:
        return {"method": "GEV_annual_maxima", "n_maxima": 0, "shape": np.nan,
                "loc": np.nan, "scale": np.nan, "return_levels": {}, "note": "insufficient data"}

    annual_max = s.groupby(s.index.year).max()
    annual_max = annual_max[annual_max.notna()]
    if len(annual_max) < 5:
        return {"method": "GEV_annual_maxima", "n_maxima": int(len(annual_max)), "shape": np.nan,
                "loc": np.nan, "scale": np.nan, "return_levels": {}, "note": "insufficient annual maxima"}

    c, loc, scale = stats.genextreme.fit(annual_max.values)
    levels = {}
    for T in return_periods_years:
        levels[T] = float(stats.genextreme.ppf(1.0 - 1.0 / T, c=c, loc=loc, scale=scale))

    return {"method": "GEV_annual_maxima", "n_maxima": int(len(annual_max)),
            "shape": float(c), "loc": float(loc), "scale": float(scale),
            "return_levels": levels, "note": ""}


def fit_pot_gpd(series: pd.Series, threshold_q=0.90, return_periods_years=(2, 5, 10, 20, 50, 100)):
    """Fit a Generalized Pareto to threshold exceedances and return fitted return levels."""
    s = series.dropna()
    n_total = len(s)
    if n_total < 30:
        return {"method": "POT_GPD", "threshold": np.nan, "threshold_q": threshold_q,
                "n_exceedances": 0, "shape": np.nan, "scale": np.nan,
                "return_levels": {}, "note": "insufficient data"}

    thr = float(np.quantile(s.values, threshold_q))
    exc = s.values[s.values > thr] - thr
    if len(exc) < 8:
        return {"method": "POT_GPD", "threshold": thr, "threshold_q": threshold_q,
                "n_exceedances": int(len(exc)), "shape": np.nan, "scale": np.nan,
                "return_levels": {}, "note": "too few exceedances"}

    c, loc, scale = stats.genpareto.fit(exc, floc=0)
    rate_per_month = len(exc) / n_total

    levels = {}
    for T in return_periods_years:
        n_months = T * MONTHS_PER_YEAR
        p_exceed = 1.0 / (n_months * rate_per_month)
        if not (0 < p_exceed < 1):
            levels[T] = np.nan
            continue
        levels[T] = float(thr + stats.genpareto.ppf(1.0 - p_exceed, c=c, loc=0, scale=scale))

    return {"method": "POT_GPD", "threshold": thr, "threshold_q": threshold_q,
            "n_exceedances": int(len(exc)), "shape": float(c), "scale": float(scale),
            "exceedance_rate_per_month": float(rate_per_month),
            "return_levels": levels, "note": ""}
