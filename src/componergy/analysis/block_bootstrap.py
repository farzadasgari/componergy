"""
Moving-block bootstrap for serially dependent monthly series.
"""

from __future__ import annotations

import numpy as np

DEFAULT_BLOCK_MONTHS = 12


def lag1_autocorrelation(values) -> float:
    """
    Lag-1 autocorrelation of a series, NaN-safe. Returns NaN for degenerate input.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan
    v = v - v.mean()
    denom = np.sum(v * v)
    if denom == 0:
        return np.nan
    return float(np.sum(v[:-1] * v[1:]) / denom)


```
