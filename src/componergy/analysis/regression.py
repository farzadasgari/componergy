"""
Event effects with seasonal and trend controls, and inference that stays valid
when event months come in a few long episodes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TREND_SPECS = ("none", "linear", "quadratic", "year_fe")
DEFAULT_HAC_LAGS = 12
DEFAULT_MIN_EVENT = 3
DEFAULT_MIN_SHIFT = 12
CI_LEVEL = 0.95
_Z = float(stats.norm.ppf(0.5 + CI_LEVEL / 2))


def year_fraction(index) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    return np.asarray(idx.year + (idx.month - 0.5) / 12.0, dtype=float)


def design_matrix(index, dummies: dict | None = None, trend: str = "linear", month_fe: bool = True):
    """Regressors: constant, month-of-year effects, trend, then event dummies."""
    if trend not in TREND_SPECS:
        raise ValueError(f"trend must be one of {TREND_SPECS}, got {trend!r}")
    idx = pd.DatetimeIndex(index)
    n = len(idx)
    cols, names = [np.ones(n)], ["const"]

    if month_fe:
        months = idx.month.values
        for m in range(2, 13):
            cols.append((months == m).astype(float))
            names.append(f"month_{m:02d}")

    if trend in ("linear", "quadratic"):
        t = (year_fraction(idx) - year_fraction(idx).mean()) / 10.0
        cols.append(t)
        names.append("trend_per_decade")
        if trend == "quadratic":
            cols.append(t ** 2)
            names.append("trend_sq")
    elif trend == "year_fe":
        years = idx.year.values
        for yr in np.unique(years)[1:]:
            cols.append((years == yr).astype(float))
            names.append(f"year_{yr}")

    for name, values in (dummies or {}).items():
        cols.append(np.asarray(values, dtype=float))
        names.append(name)
    return np.column_stack(cols), names


def newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int, small_sample: bool = True) -> np.ndarray:
    """Newey-West HAC covariance of OLS coefficients (Bartlett kernel)."""
    n, k = X.shape
    xu = X * resid[:, None]
    S = xu.T @ xu
    for j in range(1, min(int(lags), n - 1) + 1):
        w = 1.0 - j / (lags + 1.0)
        G = xu[j:].T @ xu[:-j]
        S += w * (G + G.T)
    bread = np.linalg.pinv(X.T @ X)
    cov = bread @ S @ bread
    if small_sample and n > k:
        cov = cov * (n / (n - k))
    return cov


def ols_hac(y, X: np.ndarray, names: list, lags: int = DEFAULT_HAC_LAGS, use_t: bool = True) -> dict:
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    n, k = X.shape
    rank = int(np.linalg.matrix_rank(X))
    cov = newey_west_cov(X, resid, lags)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = np.where(se > 0, beta / se, np.nan)
    df_resid = max(n - rank, 1)
    pvals = 2.0 * (stats.t.sf(np.abs(tvals), df_resid) if use_t else stats.norm.sf(np.abs(tvals)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return {"names": list(names), "beta": beta, "se": se, "t": tvals, "p": pvals, "cov": cov,
            "fitted": fitted, "resid": resid, "n": int(n), "k": int(k), "rank": rank,
            "rank_deficient": bool(rank < k), "df_resid": int(df_resid), "r2": r2,
            "hac_lags": int(lags), "use_t": bool(use_t)}


def hac_contrast(fit: dict, weights: dict) -> dict:
    """Estimate, HAC standard error and p-value of sum_j w_j * beta_j (reference)."""
    names = fit["names"]
    w = np.zeros(len(names))
    for name, value in weights.items():
        if name not in names:
            return {"estimate": np.nan, "se": np.nan, "p": np.nan}
        w[names.index(name)] = value
    est = float(w @ fit["beta"])
    se = float(np.sqrt(max(w @ fit["cov"] @ w, 0.0)))
    if se <= 0 or not np.isfinite(se):
        return {"estimate": est, "se": se, "p": np.nan}
    t = est / se
    p = float(2.0 * (stats.t.sf(abs(t), fit["df_resid"]) if fit["use_t"] else stats.norm.sf(abs(t))))
    return {"estimate": est, "se": se, "p": p}


def shift_range(n: int, min_shift: int = DEFAULT_MIN_SHIFT) -> np.ndarray:
    """Distinct circular shifts to use: at least `min_shift` months from zero in either direction."""
    if n > 2 * min_shift:
        return np.arange(min_shift, n - min_shift + 1)
    return np.arange(1, max(2, n))


def shift_null_coefficients(y: np.ndarray, Z: np.ndarray, D: np.ndarray,
                            min_shift: int = DEFAULT_MIN_SHIFT) -> np.ndarray:
    """Coefficients of the event dummies when the whole event calendar is shifted."""
    n = len(y)
    M = np.eye(n) - Z @ np.linalg.pinv(Z)
    yt = M @ y
    out = []
    for s in shift_range(n, min_shift):
        Ds = np.roll(D, int(s), axis=0)
        G = Ds.T @ M @ Ds
        b = Ds.T @ yt
        try:
            out.append(np.linalg.solve(G, b))
        except np.linalg.LinAlgError:
            out.append(np.linalg.pinv(G) @ b)
    return np.asarray(out)


def randomization_inference(estimate: float, null: np.ndarray) -> dict:
    """Standard error, CI and two p-values of `estimate` against its shift-null distribution."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    S = int(len(null))
    if S < 10 or not np.isfinite(estimate):
        return {"estimate": estimate, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "p_value": np.nan, "p_exact": np.nan, "n_shifts": S}
    se = float(null.std(ddof=1))
    p_exact = float((1 + np.sum(np.abs(null) >= abs(estimate))) / (S + 1))
    p_norm = float(2.0 * stats.norm.sf(abs(estimate - null.mean()) / se)) if se > 0 else np.nan
    return {"estimate": float(estimate), "se": se, "ci_lo": float(estimate - _Z * se),
            "ci_hi": float(estimate + _Z * se), "p_value": p_norm, "p_exact": p_exact, "n_shifts": S}


def _as_bool(s, index):
    return pd.Series(s).reindex(index).fillna(False).astype(bool)


def event_effects(y, event_masks: dict, normal_mask, trend: str = "linear",
                  hac_lags: int = DEFAULT_HAC_LAGS, month_fe: bool = True, exclusive: bool = True,
                  min_event: int = DEFAULT_MIN_EVENT, min_shift: int = DEFAULT_MIN_SHIFT) -> dict:
    """Fit the event-effect model for one series (effects in the units of `y`)."""
    from componergy.analysis.events import exclusive_event_masks

    y = pd.Series(y).astype(float)
    masks = {k: _as_bool(v, y.index) for k, v in event_masks.items()}
    if exclusive:
        masks = exclusive_event_masks(masks)
    normal = _as_bool(normal_mask, y.index)

    in_sample = normal.copy()
    for m in masks.values():
        in_sample = in_sample | m
    in_sample = in_sample & y.notna()

    ys = y[in_sample]
    dummies, counts = {}, {}
    for name, m in masks.items():
        ms = m[in_sample]
        counts[name] = int(ms.sum())
        if counts[name] >= min_event:
            dummies[name] = ms.values

    X, names = design_matrix(ys.index, dummies, trend=trend, month_fe=month_fe)
    fit = ols_hac(ys.values, X, names, lags=hac_lags)
    dnames = list(dummies)

    null = np.empty((0, 0))
    if dnames:
        Z, _ = design_matrix(ys.index, None, trend=trend, month_fe=month_fe)
        D = np.column_stack([dummies[k].astype(float) for k in dnames])
        null = shift_null_coefficients(ys.values, Z, D, min_shift=min_shift)

    X0 = X.copy()
    for name in dnames:
        X0[:, names.index(name)] = 0.0
    baseline = X0 @ fit["beta"]

    rows = []
    for name in masks:
        if name in dummies:
            j = dnames.index(name)
            est = float(fit["beta"][names.index(name)])
            ri = randomization_inference(est, null[:, j])
            hac = hac_contrast(fit, {name: 1.0})
            base_mean = float(np.mean(baseline[dummies[name]]))
            rows.append({"Event": name, "N_event": counts[name], "Delta_native": est,
                         "SE_native": ri["se"], "CI_lo_native": ri["ci_lo"], "CI_hi_native": ri["ci_hi"],
                         "tstat": est / ri["se"] if ri["se"] and np.isfinite(ri["se"]) else np.nan,
                         "p_value": ri["p_value"], "p_exact": ri["p_exact"], "p_hac": hac["p"],
                         "Baseline_mean": base_mean,
                         "Delta_pct": 100.0 * est / base_mean if base_mean else np.nan,
                         "N_shifts": ri["n_shifts"]})
        else:
            rows.append({"Event": name, "N_event": counts[name], "Delta_native": np.nan, "SE_native": np.nan,
                         "CI_lo_native": np.nan, "CI_hi_native": np.nan, "tstat": np.nan, "p_value": np.nan,
                         "p_exact": np.nan, "p_hac": np.nan, "Baseline_mean": np.nan, "Delta_pct": np.nan,
                         "N_shifts": 0})

    info = {"n_used": int(in_sample.sum()), "n_normal": int((normal & in_sample).sum()),
            "n_dropped_other": int((y.notna() & ~in_sample).sum()), "trend": trend,
            "hac_lags": int(hac_lags), "month_fe": bool(month_fe), "exclusive": bool(exclusive),
            "rank_deficient": fit["rank_deficient"], "r2": fit["r2"]}
    return {"effects": pd.DataFrame(rows), "fit": fit, "null": null, "dummy_names": dnames, "info": info}


AMPLIFICATION_WEIGHTS = {"Compound": 1.0, "Heatwave": -1.0, "Drought": -1.0}


def amplification_contrast(result: dict, weights: dict | None = None) -> dict:
    """Compound minus (heat-only + drought-only): does the compound response exceed the additive sum?"""
    weights = weights or AMPLIFICATION_WEIGHTS
    dn = result["dummy_names"]
    nan = {"contrast": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan,
           "p_exact": np.nan, "p_hac": np.nan, "additive_expectation": np.nan, "compound": np.nan,
           "n_shifts": 0}
    if any(k not in dn for k in weights):
        return nan
    fit, null = result["fit"], result["null"]
    names = fit["names"]
    w_full = {k: v for k, v in weights.items()}
    est = float(sum(v * fit["beta"][names.index(k)] for k, v in w_full.items()))
    null_c = null @ np.array([w_full.get(k, 0.0) for k in dn])
    ri = randomization_inference(est, null_c)
    hac = hac_contrast(fit, w_full)
    add = float(sum(fit["beta"][names.index(k)] for k in ("Heatwave", "Drought")))
    return {"contrast": est, "se": ri["se"], "ci_lo": ri["ci_lo"], "ci_hi": ri["ci_hi"],
            "p_value": ri["p_value"], "p_exact": ri["p_exact"], "p_hac": hac["p"],
            "additive_expectation": add, "compound": float(fit["beta"][names.index("Compound")]),
            "n_shifts": ri["n_shifts"]}


def normal_baseline_anomalies(y, normal_mask, trend: str = "linear", month_fe: bool = True):
    """Counterfactual anomalies: y minus what the NORMAL months predict for that month and year."""
    if isinstance(y, pd.DataFrame):
        parts = [normal_baseline_anomalies(y[c], normal_mask, trend, month_fe) for c in y.columns]
        anom = pd.concat([p[0].rename(c) for p, c in zip(parts, y.columns)], axis=1)
        base = pd.concat([p[1].rename(c) for p, c in zip(parts, y.columns)], axis=1)
        return anom, base

    y = pd.Series(y).astype(float)
    normal = _as_bool(normal_mask, y.index) & y.notna()
    X, _ = design_matrix(y.index, None, trend=trend, month_fe=month_fe)
    beta, *_ = np.linalg.lstsq(X[normal.values], y[normal].values, rcond=None)
    base = pd.Series(X @ beta, index=y.index)
    return (y - base), base


def baseline_adjusted_z(y, normal_mask, trend: str = "linear", month_fe: bool = True):
    """Counterfactual anomalies scaled by their own standard deviation (dimensionless)."""
    anom, _ = normal_baseline_anomalies(y, normal_mask, trend, month_fe)
    sd = anom.std(ddof=1)
    return anom / sd if isinstance(anom, pd.Series) else anom.divide(sd, axis=1)


def adjusted_composite_table(raw_df: pd.DataFrame, event_masks: dict, normal_mask, trend: str = "linear",
                             hac_lags: int = DEFAULT_HAC_LAGS, exclusive: bool = True,
                             min_shift: int = DEFAULT_MIN_SHIFT) -> pd.DataFrame:
    """Seasonally and trend-adjusted event composites, one row per (Event, Source)."""
    rows = []
    for col in raw_df.columns:
        res = event_effects(raw_df[col], event_masks, normal_mask, trend=trend, hac_lags=hac_lags,
                            exclusive=exclusive, min_shift=min_shift)
        anom, _ = normal_baseline_anomalies(raw_df[col], normal_mask, trend)
        scale = float(anom.std(ddof=1))
        for _, r in res["effects"].iterrows():
            d = float(r["Delta_native"])
            z = d / scale if scale else np.nan
            rows.append({
                "Event": r["Event"], "Source": col, "N_event": int(r["N_event"]),
                "N_normal": res["info"]["n_normal"],
                "Delta_native": d, "SE_native": r["SE_native"],
                "CI_lo_native": r["CI_lo_native"], "CI_hi_native": r["CI_hi_native"],
                "Baseline_mean": r["Baseline_mean"], "Delta_pct": r["Delta_pct"],
                "Scale": scale, "Delta_z": z,
                "CI_lo_z": r["CI_lo_native"] / scale if scale else np.nan,
                "CI_hi_z": r["CI_hi_native"] / scale if scale else np.nan,
                "Mean_normal": r["Baseline_mean"], "Mean_event": r["Baseline_mean"] + d,
                "Difference": z, "tstat": r["tstat"], "p_value": r["p_value"],
                "p_exact": r["p_exact"], "p_hac": r["p_hac"], "N_shifts": r["N_shifts"],
                "Trend": trend, "N_dropped_other": res["info"]["n_dropped_other"],
            })
    return pd.DataFrame(rows)
