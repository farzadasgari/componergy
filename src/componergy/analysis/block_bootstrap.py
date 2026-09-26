"""Moving-block bootstrap for serially dependent monthly series."""

from __future__ import annotations

import numpy as np

DEFAULT_BLOCK_MONTHS = 12


def lag1_autocorrelation(values) -> float:
    """Lag-1 autocorrelation of a series, NaN-safe. Returns NaN for degenerate input."""
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
    """Block length from the n^(1/3) rule scaled by lag-1 autocorrelation, clipped to [min_block, max_block]."""
    rho = lag1_autocorrelation(values)
    if not np.isfinite(rho) or rho <= 0:
        return min_block
    rho = min(rho, 0.99)
    n = len(np.asarray(values, dtype=float))
    block = int(np.ceil((n ** (1.0 / 3.0)) * ((2.0 * rho / (1.0 - rho ** 2)) ** (2.0 / 3.0))))
    return int(np.clip(block, min_block, max_block))


def moving_block_indices(n: int, block: int, rng) -> np.ndarray:
    """Indices for one moving-block bootstrap replicate: contiguous runs of `block`, truncated to length n."""
    if block < 1:
        block = 1
    if block >= n:
        return np.arange(n)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    return idx[:n]


def block_bootstrap_raf(values, event_mask, normal_mask, thr, block=None,
                        n_boot=5000, seed=42, eps=1e-9):
    """RAF with a moving-block bootstrap CI, using the manuscript's ratio-minus-1 convention."""
    v = np.asarray(values, dtype=float)
    ev = np.asarray(event_mask, dtype=bool)
    nm = np.asarray(normal_mask, dtype=bool)
    n = len(v)

    if block is None:
        block = suggest_block_length(v)

    p_e = float(np.mean(v[ev] > thr)) if ev.sum() else np.nan
    p_n = float(np.mean(v[nm] > thr)) if nm.sum() else np.nan
    if not (np.isfinite(p_e) and np.isfinite(p_n)):
        return {"raf": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "block": int(block),
                "p_event": p_e, "p_normal": p_n, "n_boot_valid": 0}

    raf = p_e / max(p_n, eps) - 1.0

    rng = np.random.default_rng(seed)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = moving_block_indices(n, block, rng)
        vb, eb, nb = v[idx], ev[idx], nm[idx]
        if eb.sum() < 3 or nb.sum() < 3:
            continue
        pe = np.mean(vb[eb] > thr)
        pn = np.mean(vb[nb] > thr)
        boots[i] = pe / max(pn, eps) - 1.0

    good = boots[np.isfinite(boots)]
    if len(good) == 0:
        return {"raf": float(raf), "ci_lo": np.nan, "ci_hi": np.nan, "block": int(block),
                "p_event": p_e, "p_normal": p_n, "n_boot_valid": 0}

    lo, hi = np.quantile(good, [0.025, 0.975])
    return {"raf": float(raf), "ci_lo": float(lo), "ci_hi": float(hi), "block": int(block),
            "p_event": p_e, "p_normal": p_n, "n_boot_valid": int(len(good))}
