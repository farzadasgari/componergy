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


def suggest_block_length(values, min_block=2, max_block=24) -> int:
    """
    Block length from the n^(1/3) rule scaled by lag-1 autocorrelation, clipped to [min_block, max_block].

    Returns min_block for series with no positive autocorrelation, where
    blocking buys nothing. The clip keeps the block within a range that
    still leaves many distinct blocks in a ~200-month record.
    """
    rho = lag1_autocorrelation(values)
    if not np.isfinite(rho) or rho <= 0:
        return min_block
    rho = min(rho, 0.99)
    n = len(np.asarray(values, dtype=float))
    block = int(np.ceil((n ** (1.0 / 3.0)) * ((2.0 * rho / (1.0 - rho ** 2)) ** (2.0 / 3.0))))
    return int(np.clip(block, min_block, max_block))


def moving_block_indices(n: int, block: int, rng) -> np.ndarray:
    """
    Indices for one moving-block bootstrap replicate: contiguous runs of `block`, truncated to length n.

    Blocks may overlap and repeat, which is what makes this the moving
    block bootstrap. block=1 reduces exactly to the i.i.d. bootstrap,
    which is convenient for side-by-side comparison.
    """
    if block < 1:
        block = 1
    if block >= n:
        return np.arange(n)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    return idx[:n]
