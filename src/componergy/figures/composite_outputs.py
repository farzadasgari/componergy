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


def write_composite_inference(raw_df: pd.DataFrame, event_masks: dict, normal, out_dir, prefix: str,
                              label: str = "Source", trend: str = "linear", n_boot: int = 2000) -> dict:
    """Compute and write the four tables; return them (and log-ready strings) for the figure log."""
    from componergy.netcdf_io import atomic_to_csv

    adj_z = baseline_adjusted_z(raw_df, normal, trend=trend)
    superadd = superadditivity_table(adj_z, event_masks, normal, n_boot=n_boot, label=label)
    amplification = amplification_table(raw_df, event_masks, normal, trend=trend, label=label)
    placebo = placebo_table(adj_z, exclusive_event_masks(event_masks), normal, label=label)
    robustness = specification_robustness(raw_df, event_masks, normal, label=label)

    atomic_to_csv(amplification, out_dir / f"{prefix}_amplification.csv", index=False)
    atomic_to_csv(superadd, out_dir / f"{prefix}_superadditivity.csv", index=False)
    atomic_to_csv(placebo, out_dir / f"{prefix}_placebo.csv", index=False)
    atomic_to_csv(robustness, out_dir / f"{prefix}_specification_robustness.csv", index=False)

    sections = {
        "Amplification contrast: compound - (heat-only + drought-only) "
        "(month + trend controls, calendar-shift randomization, FDR across series)":
            amplification.to_string(index=False),
        "Super-additivity on counterfactual anomalies (randomization primary; block and iid "
        "bootstrap shown for comparison)": superadd.to_string(index=False),
        "Placebo (shifted event calendar; exact p floored at 1/(n_shifts+1))": placebo.to_string(index=False),
        "Specification robustness (naive composite vs controlled estimates)": robustness.to_string(index=False),
    }
    return {"amplification": amplification, "superadditivity": superadd, "placebo": placebo,
            "robustness": robustness, "log_sections": sections}


INFERENCE_NOTE = (
    "Event effects are estimated with month-of-year effects and a trend, so an event month is compared "
    "with normal months of the SAME calendar month and year rather than with a pooled all-season mean; "
    "categories are mutually exclusive (compound months are removed from the heat-only and drought-only "
    "sets). Significance uses calendar-shift randomization: the whole event calendar is shifted "
    "circularly through the record and the model refitted, which carries autocorrelation, seasonality "
    "and the clustering of events into episodes into the null. In simulation on series with no effect "
    "this rejected 4-8% of the time at a nominal 5%, versus 14-26% for Newey-West HAC and 8-19% for a "
    "moving-block bootstrap, which are reported (p_hac) for reference only. q_value is Benjamini-"
    "Hochberg on the studentized randomization p; p_exact cannot fall below 1/(n_shifts+1)."
)
