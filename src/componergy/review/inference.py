# src/componergy/review/inference.py
"""
Alternative inference for the event and continuous regressions: season-preserving shifts, water-year
block permutation and a water-year pairs block bootstrap.

All functions are pure (arrays in, arrays/dicts out). Every null uses the Frisch-Waugh algebra of
`analysis.regression.shift_null_coefficients`: with M the residual-maker of the nuisance design Z, the
coefficients of a term block D are (D'MD)^-1 D'M y, so a null only needs the term block re-arranged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from componergy.analysis.regression import randomization_inference

SEED = 42


def season_preserving_shifts(n: int, min_shift: int = 12) -> np.ndarray:
    """Circular offsets 12, 24, ..., <= n-12. Multiples of 12 keep every month on its own calendar month."""
    if n % 12 != 0:
        raise ValueError(f"a 12-multiple circular shift only preserves calendar alignment on whole years; "
                         f"n = {n} is not a multiple of 12")
    return np.arange(min_shift, n - min_shift + 1, 12)


def whole_year_window(index, kind: str = "calendar") -> pd.DatetimeIndex:
    """Largest span of whole years inside a contiguous monthly `index`: Jan-Dec ("calendar") or Oct-Sep ("water")."""
    start_month, end_month = {"calendar": (1, 12), "water": (10, 9)}[kind]
    idx = pd.DatetimeIndex(index)
    months = idx.month.values
    first = np.flatnonzero(months == start_month)
    last = np.flatnonzero(months == end_month)
    i0 = int(first[0]) if len(first) else len(idx)
    ends = last[last >= i0 + 11]
    if len(ends) == 0:
        raise ValueError(f"no whole {kind} year inside the index")
    i1 = int(ends[-1])
    win = idx[i0:i1 + 1]
    if len(win) % 12:
        raise ValueError("index is not a contiguous monthly series")
    return win


def _coefficients(Dm: np.ndarray, M: np.ndarray, yt: np.ndarray) -> np.ndarray:
    """Coefficients (D'MD)^-1 D'M y for a stack of term blocks Dm of shape (S, n, k)."""
    MD = M @ Dm
    DT = np.swapaxes(Dm, 1, 2)
    G, b = DT @ MD, DT @ yt
    try:
        return np.linalg.solve(G, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        out = np.empty(b.shape)
        for i in range(len(G)):
            try:
                out[i] = np.linalg.solve(G[i], b[i])
            except np.linalg.LinAlgError:
                out[i] = np.linalg.pinv(G[i]) @ b[i]
        return out


def _residual_maker(y, Z, keep):
    Zk = np.asarray(Z, dtype=float)[keep]
    M = np.eye(len(Zk)) - Zk @ np.linalg.pinv(Zk)
    return M, M @ np.asarray(y, dtype=float)[keep]


def _as_block(D) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    return D[:, None] if D.ndim == 1 else D


def shift_null_generic(y, Z, D, offsets, keep=None, batch: int = 400) -> np.ndarray:
    """
    Term-block coefficients when the whole block D is circularly shifted by each of `offsets`; shape (S, k).

    Same algebra as `regression.shift_null_coefficients`, with an explicit offset list. `keep` (bool, length n)
    marks the rows that enter the fit; D is rolled over ALL n rows first, so months that were dropped from the
    sample still occupy their calendar slot (needed for calendar-preserving shifts).
    """
    D = _as_block(D)
    n = len(D)
    keep = np.ones(n, bool) if keep is None else np.asarray(keep, bool)
    M, yt = _residual_maker(y, Z, keep)
    rows = np.arange(n)
    offsets = np.asarray(offsets, dtype=int)
    out = []
    for lo in range(0, len(offsets), batch):
        idx = (rows[None, :] - offsets[lo:lo + batch, None]) % n  # np.roll(D, s)[i] = D[(i - s) % n]
        out.append(_coefficients(D[idx][:, keep, :], M, yt))
    return np.concatenate(out) if out else np.empty((0, D.shape[1]))


def year_block_permutation_null(y, Z, D, index, kind: str = "water", n_perm: int = 1999, seed: int = SEED,
                                keep=None, batch: int = 400) -> np.ndarray:
    """
    Term-block coefficients when whole (water) years of D are permuted among years; shape (n_perm, k).

    Outcome and nuisance design stay fixed. The order of the year-long D blocks is permuted at random, so
    every event keeps its calendar month and the clustering of events inside a year, while multi-year
    clustering (consecutive dry/hot years) is broken. Rows outside the whole-year window are discarded.
    """
    D = _as_block(D)
    idx = pd.DatetimeIndex(index)
    win = whole_year_window(idx, kind)
    sel = np.flatnonzero(idx.isin(win))
    keep = np.ones(len(idx), bool) if keep is None else np.asarray(keep, bool)
    Dw, keepw = D[sel], keep[sel]
    n_years = len(sel) // 12
    M, yt = _residual_maker(np.asarray(y, dtype=float)[sel], np.asarray(Z, dtype=float)[sel], keepw)
    blocks = Dw.reshape(n_years, 12, -1)
    rng = np.random.default_rng(seed)
    out = []
    for lo in range(0, n_perm, batch):
        orders = np.array([rng.permutation(n_years) for _ in range(min(batch, n_perm - lo))])
        Dp = blocks[orders].reshape(len(orders), n_years * 12, -1)
        out.append(_coefficients(Dp[:, keepw, :], M, yt))
    return np.concatenate(out) if out else np.empty((0, D.shape[1]))


def water_year_labels(index) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    return np.asarray(idx.year + (idx.month >= 10), dtype=int)


def pairs_block_bootstrap(y, X, index, coef_idx, block_years: int = 2, n_boot: int = 2000, seed: int = SEED,
                          q=(2.5, 97.5)) -> dict:
    """
    Circular block bootstrap over water years: blocks of `block_years` consecutive water years are drawn
    with replacement and (y, X) rows are resampled together. `y` may be 1-D or (n, m) (several outcomes
    sharing the same draws). Percentile 95% CI of the coefficients `coef_idx`.

    A draw is dropped when, after removing all-zero columns (an absent year dummy), the design is rank
    deficient or any of `coef_idx` is not estimable (an event dummy that is zero everywhere).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    coef_idx = list(coef_idx)
    labels = water_year_labels(index)
    years = np.unique(labels)
    W = len(years)
    by_year = [np.flatnonzero(labels == yr) for yr in years]
    n_blocks = int(np.ceil(W / block_years))
    rng = np.random.default_rng(seed)
    draws, dropped = [], 0
    for _ in range(int(n_boot)):
        starts = rng.integers(0, W, size=n_blocks)
        rows = np.concatenate([by_year[(s + j) % W] for s in starts for j in range(block_years)])
        Xb = X[rows]
        present = np.any(Xb != 0.0, axis=0)
        if not present[coef_idx].all():
            dropped += 1
            continue
        Xr = Xb[:, present]
        if np.linalg.matrix_rank(Xr) < Xr.shape[1]:
            dropped += 1
            continue
        beta = np.linalg.lstsq(Xr, y[rows], rcond=None)[0]
        pos = np.cumsum(present) - 1
        draws.append(beta[pos[coef_idx]])
    draws = np.asarray(draws)
    if len(draws) >= 20:
        lo, hi = np.percentile(draws, q, axis=0)
    else:
        lo = hi = np.full(draws.shape[1:] if len(draws) else (len(coef_idx),), np.nan)
    return {"draws": draws, "ci_lo": lo, "ci_hi": hi, "n_ok": int(len(draws)), "n_dropped": int(dropped),
            "n_years": int(W), "block_years": int(block_years)}


def p_from_null(est: float, null) -> dict:
    """Normal-approximation p (studentized), exact p (1+#|null|>=|est|)/(S+1), null SE and size."""
    ri = randomization_inference(est, null)
    return {"p_norm": ri["p_value"], "p_exact": ri["p_exact"], "se": ri["se"], "n_null": ri["n_shifts"]}
