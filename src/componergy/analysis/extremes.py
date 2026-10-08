"""
Extreme-value analysis of the systemic risk series: empirical plotting
positions plus two fitted extreme-value models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from componergy.analysis.block_bootstrap import moving_block_indices

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


def decluster_runs(series: pd.Series, threshold: float, run_length: int = 3) -> pd.Series:
    """Runs declustering: one value per cluster of exceedances."""
    s = series.dropna()
    clusters, current, gap = [], [], 0
    for t, v in zip(s.index, s.values):
        if v > threshold:
            current.append((t, v))
            gap = 0
        elif current:
            gap += 1
            if gap >= run_length:
                clusters.append(max(current, key=lambda x: x[1]))
                current, gap = [], 0
    if current:
        clusters.append(max(current, key=lambda x: x[1]))
    return pd.Series([v for _, v in clusters], index=pd.DatetimeIndex([t for t, _ in clusters]), dtype=float)


def fit_pot_gpd(series: pd.Series, threshold_q=0.90, return_periods_years=(2, 5, 10, 20, 50, 100),
                decluster_run=None):
    """Fit a Generalized Pareto to threshold exceedances and return fitted return levels."""
    s = series.dropna()
    n_total = len(s)
    base = {"method": "POT_GPD" + (f"_declustered_r{decluster_run}" if decluster_run else ""),
            "threshold_q": threshold_q, "declustered": bool(decluster_run)}
    if n_total < 30:
        return {**base, "threshold": np.nan, "n_exceedances": 0, "shape": np.nan, "scale": np.nan,
                "return_levels": {}, "note": "insufficient data"}

    thr = float(np.quantile(s.values, threshold_q))
    if decluster_run:
        exc_values = decluster_runs(s, thr, decluster_run).values
    else:
        exc_values = s.values[s.values > thr]
    exc = exc_values - thr
    if len(exc) < 8:
        return {**base, "threshold": thr, "n_exceedances": int(len(exc)), "shape": np.nan, "scale": np.nan,
                "return_levels": {}, "note": "too few exceedances"}

    c_, _, scale = stats.genpareto.fit(exc, floc=0)
    rate_per_month = len(exc) / n_total

    levels = {}
    for T in return_periods_years:
        p_exceed = 1.0 / (T * MONTHS_PER_YEAR * rate_per_month)
        levels[T] = float(thr + stats.genpareto.ppf(1.0 - p_exceed, c=c_, loc=0, scale=scale)) \
            if 0 < p_exceed < 1 else np.nan

    return {**base, "threshold": thr, "n_exceedances": int(len(exc)), "shape": float(c_),
            "scale": float(scale), "exceedance_rate_per_month": float(rate_per_month),
            "return_levels": levels, "note": ""}


def return_period_of_value(value, gev_fit=None, pot_fit=None):
    out = {"value": float(value), "gev_return_period_years": np.nan, "pot_return_period_years": np.nan}

    if gev_fit and np.isfinite(gev_fit.get("shape", np.nan)):
        cdf = stats.genextreme.cdf(value, c=gev_fit["shape"], loc=gev_fit["loc"], scale=gev_fit["scale"])
        if cdf < 1.0:
            out["gev_return_period_years"] = float(1.0 / (1.0 - cdf))

    if pot_fit and np.isfinite(pot_fit.get("shape", np.nan)) and value > pot_fit["threshold"]:
        surv = stats.genpareto.sf(value - pot_fit["threshold"], c=pot_fit["shape"], loc=0, scale=pot_fit["scale"])
        rate = pot_fit["exceedance_rate_per_month"]
        if surv > 0 and rate > 0:
            out["pot_return_period_years"] = float(1.0 / (surv * rate * MONTHS_PER_YEAR))

    return out


def bootstrap_return_period(series: pd.Series, value: float, n_boot: int = 500, block: int = 12,
                            seed: int = 42, threshold_q: float = 0.90, decluster_run: int | None = 3) -> dict:
    rng = np.random.default_rng(seed)
    s = series.dropna()
    n = len(s)
    gev_rp, pot_rp = [], []
    for _ in range(n_boot):
        idx = moving_block_indices(n, block, rng)
        sb = pd.Series(s.values[idx], index=s.index)
        gev = fit_gev_annual_maxima(sb)
        pot = fit_pot_gpd(sb, threshold_q=threshold_q, decluster_run=decluster_run)
        rp = return_period_of_value(value, gev, pot)
        gev_rp.append(rp["gev_return_period_years"])
        pot_rp.append(rp["pot_return_period_years"])

    def summarize(x):
        x = np.asarray(x, dtype=float)
        ok = x[np.isfinite(x)]
        if len(ok) < 20:
            return {"n_valid": int(len(ok)), "p2_5": np.nan, "median": np.nan, "p97_5": np.nan}
        lo, med, hi = np.quantile(ok, [0.025, 0.5, 0.975])
        return {"n_valid": int(len(ok)), "p2_5": float(lo), "median": float(med), "p97_5": float(hi)}

    return {"n_boot": int(n_boot), "block_months": int(block),
            "gev": summarize(gev_rp), "pot_declustered": summarize(pot_rp)}


def gev_goodness_of_fit(series: pd.Series) -> dict:
    fit = fit_gev_annual_maxima(series)
    if fit["n_maxima"] < 8 or not np.isfinite(fit["shape"]):
        return {"n_maxima": fit["n_maxima"], "ks_statistic": np.nan, "ks_p_value_descriptive": np.nan}
    s = series.dropna()
    maxima = s.groupby(s.index.year).max().values
    res = stats.kstest(maxima, lambda x: stats.genextreme.cdf(x, c=fit["shape"], loc=fit["loc"], scale=fit["scale"]))
    return {"n_maxima": int(len(maxima)), "ks_statistic": float(res.statistic),
            "ks_p_value_descriptive": float(res.pvalue)}


def threshold_stability(series: pd.Series, quantiles=(0.80, 0.85, 0.90, 0.95), return_period_years: int = 20,
                        decluster_run: int | None = 3) -> pd.DataFrame:
    rows = []
    for q in quantiles:
        fit = fit_pot_gpd(series, threshold_q=q, return_periods_years=(return_period_years,),
                          decluster_run=decluster_run)
        rows.append({"threshold_q": q, "threshold": fit.get("threshold"), "n_exceedances": fit.get("n_exceedances"),
                     "shape": fit.get("shape"), "scale": fit.get("scale"),
                     f"return_level_{return_period_years}yr": fit["return_levels"].get(return_period_years, np.nan)
                     if fit["return_levels"] else np.nan})
    return pd.DataFrame(rows)


def stationarity_check(series: pd.Series, threshold_q: float = 0.90) -> dict:
    s = series.dropna()
    years = s.index.year
    annual_max = s.groupby(years).max()
    thr = float(np.quantile(s.values, threshold_q))
    counts = (s > thr).groupby(years).sum()
    out = {"n_years": int(len(annual_max)), "threshold": thr}
    for name, ser in (("annual_maxima", annual_max), ("annual_exceedance_count", counts)):
        if len(ser) >= 8:
            tau, p = stats.kendalltau(ser.index.values, ser.values)
            out[f"{name}_kendall_tau"] = float(tau)
            out[f"{name}_trend_p_value"] = float(p)
        else:
            out[f"{name}_kendall_tau"] = out[f"{name}_trend_p_value"] = np.nan
    return out
