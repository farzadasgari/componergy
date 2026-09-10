"""
Gringorten plotting-position standardization.

Used for the final SCDHI remap step: transforming the copula joint
probability p = P(X<=x, Y>=y) into a standard-normal SCDHI value.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def gringorten_standardize(x: np.ndarray) -> np.ndarray:
    """
    Rank-based empirical standardization via the Gringorten formula.

    Requires at least 10 valid (non-NaN) values; returns all-NaN otherwise.
    Preserves the rank order of the input.
    """
    x = np.asarray(x, dtype=float)
    valid = ~np.isnan(x)
    if valid.sum() < 10:
        return np.full_like(x, np.nan)
    ranks = rankdata(x[valid], method="average")
    n = valid.sum()
    p = (ranks - 0.44) / (n + 0.12)
    p = np.clip(p, 1e-8, 1 - 1e-8)
    out = np.full_like(x, np.nan)
    out[valid] = norm.ppf(p)
    return out
