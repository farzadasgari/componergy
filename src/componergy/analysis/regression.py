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
