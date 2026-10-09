"""Relative importance of correlated predictors: LMG ("general dominance") decomposition."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from componergy.analysis.block_bootstrap import moving_block_indices
from componergy.analysis.lag_scan import block_permute


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    if X.shape[1] == 0:
        return 0.0
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(resid @ resid) / sst if sst > 0 else 0.0


def lmg_decomposition(y, X: np.ndarray) -> dict:
    """LMG shares (in R^2 units) of each column of X, exact over all orderings."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    cache = {}

    def r2_of(cols):
        key = tuple(sorted(cols))
        if key not in cache:
            cache[key] = _r2(y, X[:, list(key)])
        return cache[key]

    contrib = np.zeros(p)
    orders = list(permutations(range(p)))
    for order in orders:
        entered = []
        for j in order:
            contrib[j] += r2_of(entered + [j]) - r2_of(entered)
            entered.append(j)
    contrib /= len(orders)
    return {"lmg": contrib, "total_r2": r2_of(range(p))}


def variance_inflation(X: np.ndarray) -> np.ndarray:
    """VIF_j = 1 / (1 - R^2 of column j regressed on the others)."""
    X = np.asarray(X, dtype=float)
    out = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        r2 = _r2(X[:, j], others) if others.shape[1] else 0.0
        out.append(1.0 / max(1.0 - r2, 1e-12))
    return np.array(out)
