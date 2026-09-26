"""Validate that (demand - generation) actually behaves like net imports."""

from __future__ import annotations

import numpy as np
import pandas as pd

CA_NET_IMPORT_FRACTION_RANGE = (0.15, 0.40)


def compute_import_fraction(demand_gwh: pd.Series, generation_gwh: pd.Series) -> pd.DataFrame:
    """Per-month implied net imports and the implied import fraction of demand."""
    common = demand_gwh.index.intersection(generation_gwh.index)
    d = demand_gwh.loc[common].astype(float)
    g = generation_gwh.loc[common].astype(float)

    implied_net_import = d - g
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = implied_net_import / d.replace(0.0, np.nan)

    return pd.DataFrame({
        "Demand_GWh": d,
        "Generation_GWh": g,
        "Implied_Net_Import_GWh": implied_net_import,
        "Implied_Import_Fraction": frac,
    })

def validate_system_boundary(demand_gwh: pd.Series, generation_gwh: pd.Series,
                              expected_range=CA_NET_IMPORT_FRACTION_RANGE) -> dict:
    """Summarize the implied import fraction and judge it against the expected range."""
    df = compute_import_fraction(demand_gwh, generation_gwh)
    frac = df["Implied_Import_Fraction"].replace([np.inf, -np.inf], np.nan).dropna()

    if len(frac) == 0:
        return {"n_months": 0, "median_import_fraction": np.nan, "mean_import_fraction": np.nan,
                "p10_import_fraction": np.nan, "p90_import_fraction": np.nan,
                "fraction_of_months_negative": np.nan, "expected_range": list(expected_range),
                "median_within_expected_range": False, "verdict": "no overlapping data"}

    annual = df.groupby(df.index.year)["Implied_Import_Fraction"].median()
    median = float(frac.median())
    lo, hi = expected_range
    within = bool(lo <= median <= hi)

    if within:
        verdict = ("median implied net-import fraction falls within the expected "
                   "range for California; treating demand - generation as net "
                   "imports is consistent with published import shares")
    elif median < lo:
        verdict = ("median implied net-import fraction is BELOW the expected "
                   "range: demand and generation are closer together than "
                   "California's known import dependence implies, so the two "
                   "series may not cover the same system boundary (e.g. "
                   "generation includes output not serving in-state retail "
                   "demand, or demand excludes some load)")
    else:
        verdict = ("median implied net-import fraction is ABOVE the expected "
                   "range: the gap is wider than California's known import "
                   "dependence, so the difference likely absorbs more than net "
                   "imports alone (e.g. missing generation categories, "
                   "storage round-trip losses, or transmission losses)")

    return {
        "n_months": int(len(frac)),
        "median_import_fraction": median,
        "mean_import_fraction": float(frac.mean()),
        "p10_import_fraction": float(frac.quantile(0.10)),
        "p90_import_fraction": float(frac.quantile(0.90)),
        "fraction_of_months_negative": float((frac < 0).mean()),
        "annual_median_import_fraction": {int(y): float(v) for y, v in annual.dropna().items()},
        "expected_range": list(expected_range),
        "median_within_expected_range": within,
        "verdict": verdict,
    }
