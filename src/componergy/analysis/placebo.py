"""Placebo test for confounding in an event composite."""

from __future__ import annotations

import numpy as np
import pandas as pd

from componergy.analysis.regression import DEFAULT_MIN_SHIFT, randomization_inference, shift_range

DEFAULT_N_PERM = None
DEFAULT_BLOCK_MONTHS = DEFAULT_MIN_SHIFT
MIN_SAMPLE = 3


def circular_shift_mask(mask: np.ndarray, shift: int) -> np.ndarray:
    """Roll an event mask through the record, wrapping at the ends."""
    return np.roll(np.asarray(mask, dtype=bool), shift)


def placebo_test(values, event_mask, normal_mask, n_perm=DEFAULT_N_PERM, seed=42,
                 min_shift=DEFAULT_BLOCK_MONTHS):
    """Observed composite vs the distribution obtained by shifting the event calendar."""
    v = np.asarray(values, dtype=float)
    ev = np.asarray(event_mask, dtype=bool)
    nm = np.asarray(normal_mask, dtype=bool)
    n = len(v)
    other = ~(ev | nm)

    def delta(e_mask, n_mask):
        a, b = v[e_mask], v[n_mask]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < MIN_SAMPLE or len(b) < MIN_SAMPLE:
            return np.nan
        return float(np.mean(a) - np.mean(b))

    nan = {"observed_delta_z": np.nan, "placebo_mean": np.nan, "placebo_sd": np.nan, "p_value": np.nan,
           "p_exact": np.nan, "n_perm_valid": 0, "placebo_ci_lo": np.nan, "placebo_ci_hi": np.nan}
    observed = delta(ev, nm)
    if not np.isfinite(observed):
        return nan

    shifts = shift_range(n, min_shift)
    if n_perm is not None and n_perm < len(shifts):
        shifts = np.sort(np.random.default_rng(seed).choice(shifts, size=int(n_perm), replace=False))

    null = np.array([delta(np.roll(ev, int(s)), ~np.roll(ev, int(s)) & ~other) for s in shifts])
    null = null[np.isfinite(null)]
    if len(null) < 10:
        return {**nan, "observed_delta_z": observed}

    ri = randomization_inference(observed, null)
    lo, hi = np.quantile(null, [0.025, 0.975])
    return {"observed_delta_z": observed, "placebo_mean": float(null.mean()), "placebo_sd": ri["se"],
            "placebo_ci_lo": float(lo), "placebo_ci_hi": float(hi), "p_value": ri["p_value"],
            "p_exact": ri["p_exact"], "n_perm_valid": int(len(null))}


def placebo_table(df, event_masks, normal_mask, n_perm=DEFAULT_N_PERM, seed=42, label="Source"):
    """Run placebo_test for every (column, event) pair, FDR-corrected across all of them."""
    from componergy.figures.composite_stats import benjamini_hochberg_fdr

    normal = normal_mask.reindex(df.index).fillna(False).values
    rows = []
    for i, col in enumerate(df.columns):
        values = df[col].values
        for j, (event_name, mask) in enumerate(sorted(event_masks.items())):
            ev = mask.reindex(df.index).fillna(False).values
            res = placebo_test(values, ev, normal, n_perm=n_perm, seed=seed + i * 10 + j)
            rows.append({label: col, "Event": event_name, **res})

    out = pd.DataFrame(rows)
    if len(out):
        out["q_value"] = benjamini_hochberg_fdr(out["p_value"].values)
    return out
