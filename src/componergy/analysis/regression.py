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
