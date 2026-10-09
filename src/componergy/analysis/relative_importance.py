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


def lmg_table(y: pd.Series, X: pd.DataFrame, n_boot: int = 300, block: int = 12, seed: int = 42) -> pd.DataFrame:
    """LMG shares with indicative block-bootstrap intervals and VIFs, one row per predictor."""
    df = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    names = list(X.columns)
    if len(df) < 24:
        return pd.DataFrame(columns=["Predictor", "LMG_R2", "LMG_share", "CI_lo_share", "CI_hi_share", "VIF",
                                     "Total_R2", "n"])
    yv, Xv = df["__y__"].values, df[names].values
    base = lmg_decomposition(yv, Xv)
    vif = variance_inflation(Xv)

    rng = np.random.default_rng(seed)
    shares = []
    for _ in range(n_boot):
        idx = moving_block_indices(len(df), block, rng)
        d = lmg_decomposition(yv[idx], Xv[idx])
        if d["total_r2"] > 1e-9:
            shares.append(d["lmg"] / d["total_r2"])
    shares = np.array(shares)

    rows = []
    for j, name in enumerate(names):
        share = base["lmg"][j] / base["total_r2"] if base["total_r2"] > 1e-12 else np.nan
        lo, hi = (np.quantile(shares[:, j], [0.025, 0.975]) if len(shares) >= 20 else (np.nan, np.nan))
        rows.append({"Predictor": name, "LMG_R2": float(base["lmg"][j]), "LMG_share": float(share),
                     "CI_lo_share": float(lo), "CI_hi_share": float(hi), "VIF": float(vif[j]),
                     "Total_R2": float(base["total_r2"]), "n": int(len(df))})
    return pd.DataFrame(rows)


def incremental_r2_block_permutation(y: pd.Series, base: pd.Series, add: pd.Series,
                                     n_perm: int = 1999, block: int = 24, seed: int = 42) -> dict:
    """Does `add` explain y beyond `base`? Increase in R² vs a null that permutes `add` in blocks."""
    df = pd.concat([y.rename("y"), base.rename("b"), add.rename("a")], axis=1).dropna()
    if len(df) < 24:
        return {"delta_r2": np.nan, "p_value": np.nan, "n": int(len(df)), "n_perm": 0}
    yv, bv, av = df["y"].values, df["b"].values[:, None], df["a"].values
    observed = _r2(yv, np.column_stack([bv, av])) - _r2(yv, bv)
    rng = np.random.default_rng(seed)
    null = np.array([_r2(yv, np.column_stack([bv, block_permute(av, block, rng)])) - _r2(yv, bv)
                     for _ in range(int(n_perm))])
    return {"delta_r2": float(observed), "p_value": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
            "n": int(len(df)), "n_perm": int(len(null))}
