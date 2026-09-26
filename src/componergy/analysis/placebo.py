"""Placebo (permutation) test for confounding in event-composite results."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_N_PERM = 2000
DEFAULT_BLOCK_MONTHS = 12


def circular_shift_mask(mask: np.ndarray, shift: int) -> np.ndarray:
    """Roll an event mask through the record, wrapping at the ends."""
    return np.roll(np.asarray(mask, dtype=bool), shift)


def placebo_test(values, event_mask, normal_mask, n_perm=DEFAULT_N_PERM,
                 seed=42, min_shift=DEFAULT_BLOCK_MONTHS):
    """Compare the observed composite against a null built from circularly shifted event timing."""
    v = np.asarray(values, dtype=float)
    ev = np.asarray(event_mask, dtype=bool)
    nm = np.asarray(normal_mask, dtype=bool)
    n = len(v)

    def delta_z(e_mask, n_mask):
        a = v[e_mask]
        b = v[n_mask]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            return np.nan
        return float(np.mean(a) - np.mean(b))

    observed = delta_z(ev, nm)
    if not np.isfinite(observed):
        return {"observed_delta_z": np.nan, "placebo_mean": np.nan, "placebo_sd": np.nan,
                "p_value": np.nan, "n_perm_valid": 0,
                "placebo_ci_lo": np.nan, "placebo_ci_hi": np.nan}

    rng = np.random.default_rng(seed)
    valid_shifts = np.arange(min_shift, n - min_shift) if n > 2 * min_shift else np.arange(1, max(2, n))
    null = np.full(n_perm, np.nan)
    for i in range(n_perm):
        shift = int(rng.choice(valid_shifts))
        shifted_ev = circular_shift_mask(ev, shift)
        shifted_nm = ~shifted_ev
        null[i] = delta_z(shifted_ev, shifted_nm)

    good = null[np.isfinite(null)]
    if len(good) == 0:
        return {"observed_delta_z": observed, "placebo_mean": np.nan, "placebo_sd": np.nan,
                "p_value": np.nan, "n_perm_valid": 0,
                "placebo_ci_lo": np.nan, "placebo_ci_hi": np.nan}

    p = float((np.sum(np.abs(good) >= abs(observed)) + 1) / (len(good) + 1))
    lo, hi = np.quantile(good, [0.025, 0.975])

    return {
        "observed_delta_z": observed,
        "placebo_mean": float(np.mean(good)),
        "placebo_sd": float(np.std(good, ddof=1)),
        "placebo_ci_lo": float(lo),
        "placebo_ci_hi": float(hi),
        "p_value": p,
        "n_perm_valid": int(len(good)),
    }
