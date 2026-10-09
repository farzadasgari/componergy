"""Formal test for super-additivity ("nonlinear amplification") of compound events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from componergy.analysis.block_bootstrap import moving_block_indices
from componergy.analysis.regression import DEFAULT_MIN_SHIFT, randomization_inference, shift_range

MIN_SAMPLE = 3
DEFAULT_BLOCK = 12


def _delta_z(values, event_mask, normal_mask):
    """Mean anomaly during event months minus mean during normal months; NaN if either sample is too small."""
    ev, nm = values[event_mask], values[normal_mask]
    ev, nm = ev[np.isfinite(ev)], nm[np.isfinite(nm)]
    if len(ev) < MIN_SAMPLE or len(nm) < MIN_SAMPLE:
        return np.nan
    return float(np.mean(ev) - np.mean(nm))


def _interaction(v, h, d, c, nm):
    if min(h.sum(), d.sum(), c.sum(), nm.sum()) < MIN_SAMPLE:
        return np.nan
    mn = np.mean(v[nm])
    return float((np.mean(v[c]) - mn) - ((np.mean(v[h]) - mn) + (np.mean(v[d]) - mn)))


def _percentile_ci_and_p(boots):
    boots = boots[np.isfinite(boots)]
    if len(boots) < 20:
        return np.nan, np.nan, np.nan
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = min(1.0, 2.0 * min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0))))
    return float(lo), float(hi), float(p)


def _block_bootstrap(v, h, d, c, nm, block, n_boot, rng):
    n = len(v)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = moving_block_indices(n, block, rng)
        boots[i] = _interaction(v[idx], h[idx], d[idx], c[idx], nm[idx])
    return boots


def _iid_bootstrap(v, h, d, c, nm, n_boot, rng):
    idx_h, idx_d, idx_c, idx_n = (np.where(m)[0] for m in (h, d, c, nm))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        bh = rng.choice(idx_h, size=len(idx_h), replace=True)
        bd = rng.choice(idx_d, size=len(idx_d), replace=True)
        bc = rng.choice(idx_c, size=len(idx_c), replace=True)
        bn = rng.choice(idx_n, size=len(idx_n), replace=True)
        mn = np.mean(v[bn])
        boots[i] = (np.mean(v[bc]) - mn) - ((np.mean(v[bh]) - mn) + (np.mean(v[bd]) - mn))
    return boots


def superadditivity_test(values, heat_mask, drought_mask, compound_mask, normal_mask,
                         n_boot=2000, seed=42, block=None, min_shift=DEFAULT_MIN_SHIFT, exclusive=True):
    """Test whether the compound anomaly exceeds the sum of the heat-only and drought-only anomalies."""
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    h = np.asarray(heat_mask, dtype=bool) & finite
    d = np.asarray(drought_mask, dtype=bool) & finite
    c = np.asarray(compound_mask, dtype=bool) & finite
    nm = np.asarray(normal_mask, dtype=bool) & finite
    if exclusive:
        h, d = h & ~c, d & ~c
    block = DEFAULT_BLOCK if block is None else int(block)

    dz_h, dz_d, dz_c = (_delta_z(v, m, nm) for m in (h, d, c))
    nan = {"interaction": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan, "p_exact": np.nan,
           "se_null": np.nan, "n_shifts": 0, "dz_heat": dz_h, "dz_drought": dz_d, "dz_compound": dz_c,
           "additive_expectation": np.nan, "ci_lo_block": np.nan, "ci_hi_block": np.nan, "p_block": np.nan,
           "ci_lo_iid": np.nan, "ci_hi_iid": np.nan, "p_iid": np.nan, "block_length": block,
           "n_boot_valid": 0, "exclusive": bool(exclusive)}
    if not all(np.isfinite([dz_h, dz_d, dz_c])):
        return nan

    observed = _interaction(v, h, d, c, nm)

    other = finite & ~(h | d | c | nm)
    null = []
    for s in shift_range(len(v), min_shift):
        hs, ds, cs = (np.roll(m, int(s)) for m in (h, d, c))
        null.append(_interaction(v, hs, ds, cs, finite & ~(hs | ds | cs) & ~other))
    ri = randomization_inference(observed, np.array(null))

    rng = np.random.default_rng(seed)
    boot_b = _block_bootstrap(v, h, d, c, nm, block, n_boot, rng)
    lo_b, hi_b, p_b = _percentile_ci_and_p(boot_b)
    lo_i, hi_i, p_i = _percentile_ci_and_p(_iid_bootstrap(v, h, d, c, nm, n_boot, rng))

    return {"interaction": observed, "ci_lo": ri["ci_lo"], "ci_hi": ri["ci_hi"], "p_value": ri["p_value"],
            "p_exact": ri["p_exact"], "se_null": ri["se"], "n_shifts": ri["n_shifts"],
            "dz_heat": dz_h, "dz_drought": dz_d, "dz_compound": dz_c, "additive_expectation": dz_h + dz_d,
            "ci_lo_block": lo_b, "ci_hi_block": hi_b, "p_block": p_b,
            "ci_lo_iid": lo_i, "ci_hi_iid": hi_i, "p_iid": p_i, "block_length": block,
            "n_boot_valid": int(np.isfinite(boot_b).sum()), "exclusive": bool(exclusive)}


def superadditivity_table(df, event_masks, normal_mask, n_boot=2000, seed=42, label="Source",
                          block=None, exclusive=True):
    """Run superadditivity_test on every column of `df`, FDR-corrected across columns."""
    from componergy.figures.composite_stats import benjamini_hochberg_fdr

    heat = event_masks["Heatwave"].reindex(df.index).fillna(False).values
    drought = event_masks["Drought"].reindex(df.index).fillna(False).values
    compound = event_masks["Compound"].reindex(df.index).fillna(False).values
    normal = normal_mask.reindex(df.index).fillna(False).values

    rows = []
    for i, col in enumerate(df.columns):
        res = superadditivity_test(df[col].values, heat, drought, compound, normal, n_boot=n_boot,
                                   seed=seed + i, block=block, exclusive=exclusive)
        rows.append({label: col, **res})

    out = pd.DataFrame(rows)
    if len(out):
        out["q_value"] = benjamini_hochberg_fdr(out["p_value"].values)
    return out
