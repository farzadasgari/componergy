"""
Formal test for super-additivity (nonlinear amplification) of
compound drought-heat events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SAMPLE = 3


def _delta_z(values, event_mask, normal_mask):
    """Mean anomaly during event months minus mean during normal months; NaN if either sample is too small."""
    ev = values[event_mask]
    nm = values[normal_mask]
    ev = ev[np.isfinite(ev)]
    nm = nm[np.isfinite(nm)]
    if len(ev) < MIN_SAMPLE or len(nm) < MIN_SAMPLE:
        return np.nan
    return float(np.mean(ev) - np.mean(nm))


def superadditivity_test(values, heat_mask, drought_mask, compound_mask, normal_mask,
                         n_boot=5000, seed=42):
    """Bootstrap test of whether the compound anomaly exceeds the sum of the individual anomalies."""
    v = np.asarray(values, dtype=float)
    h = np.asarray(heat_mask, dtype=bool)
    d = np.asarray(drought_mask, dtype=bool)
    c = np.asarray(compound_mask, dtype=bool)
    n = np.asarray(normal_mask, dtype=bool)

    dz_h = _delta_z(v, h, n)
    dz_d = _delta_z(v, d, n)
    dz_c = _delta_z(v, c, n)

    nan_result = {
        "interaction": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan,
        "dz_heat": dz_h, "dz_drought": dz_d, "dz_compound": dz_c,
        "additive_expectation": np.nan, "n_boot": 0,
    }
    if not all(np.isfinite([dz_h, dz_d, dz_c])):
        return nan_result

    interaction = dz_c - (dz_h + dz_d)

    rng = np.random.default_rng(seed)
    idx_h = np.where(h)[0]
    idx_d = np.where(d)[0]
    idx_c = np.where(c)[0]
    idx_n = np.where(n)[0]

    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        bh = rng.choice(idx_h, size=len(idx_h), replace=True)
        bd = rng.choice(idx_d, size=len(idx_d), replace=True)
        bc = rng.choice(idx_c, size=len(idx_c), replace=True)
        bn = rng.choice(idx_n, size=len(idx_n), replace=True)
        mn = np.mean(v[bn])
        boots[i] = (np.mean(v[bc]) - mn) - ((np.mean(v[bh]) - mn) + (np.mean(v[bd]) - mn))

    good = boots[np.isfinite(boots)]
    if len(good) == 0:
        return nan_result

    lo, hi = np.quantile(good, [0.025, 0.975])
    p = min(1.0, 2.0 * min(float(np.mean(good <= 0)), float(np.mean(good >= 0))))

    return {
        "interaction": float(interaction), "ci_lo": float(lo), "ci_hi": float(hi),
        "p_value": float(p), "dz_heat": dz_h, "dz_drought": dz_d, "dz_compound": dz_c,
        "additive_expectation": float(dz_h + dz_d), "n_boot": int(len(good)),
    }
