"""One place that writes the inference tables every event-composite figure needs."""

from __future__ import annotations

import pandas as pd

from componergy.analysis.events import exclusive_event_masks
from componergy.analysis.placebo import placebo_table
from componergy.analysis.regression import (
    adjusted_composite_table, amplification_table, baseline_adjusted_z,
)
from componergy.analysis.superadditivity import superadditivity_table

ROBUSTNESS_TRENDS = ("none", "linear", "quadratic", "year_fe")
NAIVE_LABEL = "naive composite (all normal months, Welch t; the original method)"


def specification_robustness(raw_df: pd.DataFrame, event_masks: dict, normal, label: str = "Source") -> pd.DataFrame:
    """Every event effect under the naive composite and under each control specification."""
    from componergy.figures.composite_stats import compute_composite_table

    naive = compute_composite_table(raw_df, event_masks, normal)
    rows = [pd.DataFrame({
        "Specification": NAIVE_LABEL, label: naive["Source"], "Event": naive["Event"],
        "N_event": naive["N_event"], "Delta_native": naive["Difference"], "p_value": naive["p_value"],
    })]
    for spec in ROBUSTNESS_TRENDS:
        t = adjusted_composite_table(raw_df, event_masks, normal, trend=spec)
        label_spec = {"none": "month effects only", "linear": "month effects + linear trend",
                      "quadratic": "month effects + quadratic trend",
                      "year_fe": "month + year fixed effects"}[spec]
        rows.append(pd.DataFrame({
            "Specification": label_spec, label: t["Source"], "Event": t["Event"], "N_event": t["N_event"],
            "Delta_native": t["Delta_native"], "CI_lo_native": t["CI_lo_native"],
            "CI_hi_native": t["CI_hi_native"], "Delta_pct": t["Delta_pct"], "Delta_z": t["Delta_z"],
            "p_value": t["p_value"], "p_exact": t["p_exact"], "p_hac": t["p_hac"],
        }))
    return pd.concat(rows, ignore_index=True)
