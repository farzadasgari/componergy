"""
Robustness of the compound effect and the amplification contrast to the event baseline and smoothing.

Rows are built from the existing estimators (adjusted_composite_table, amplification_table); nothing
here re-estimates anything. One row per outcome: the compound effect and the amplification contrast
A = compound - (heat-only + drought-only), with the 80%-power minimum detectable |A| at alpha = 0.05.
"""

from __future__ import annotations

import pandas as pd

from componergy.analysis.regression import adjusted_composite_table, amplification_table
from componergy.figures.composite_stats import benjamini_hochberg_fdr

BASELINES = ("full_record", "window")
SMOOTHINGS = (1, 3)
TREND = "linear"
MDE80_FACTOR = 2.8016  # z_{0.975} + z_{0.80}: two-sided alpha = 0.05, power = 0.80
COLUMNS = ("baseline", "smoothing", "family", "outcome", "n_heat", "n_drought", "n_compound",
           "n_compound_exclusive", "effect_pct", "effect_z", "p_exact", "q_bh", "A", "A_ci_lo", "A_ci_hi",
           "A_p", "A_q", "SE_A", "MDE80")


def family_rows(raw_df: pd.DataFrame, event_masks: dict, normal, meta: dict, family: str,
                baseline: str, smoothing: int, trend: str = TREND) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robustness rows for one outcome family, and the calendar-shift count behind each outcome's p-values."""
    comp = adjusted_composite_table(raw_df, event_masks, normal, trend=trend)
    comp["q_value"] = benjamini_hochberg_fdr(comp["p_value"].values)  # within family, all outcome x event
    amp = amplification_table(raw_df, event_masks, normal, trend=trend, label="outcome").set_index("outcome")
    counts = meta["counts_raw"]

    rows, shifts = [], []
    for col in raw_df.columns:
        c = comp[(comp["Source"] == col) & (comp["Event"] == "Compound")].iloc[0]
        a = amp.loc[col]
        rows.append({
            "baseline": baseline, "smoothing": int(smoothing), "family": family, "outcome": col,
            "n_heat": counts["Heatwave"], "n_drought": counts["Drought"], "n_compound": counts["Compound"],
            "n_compound_exclusive": int(c["N_event"]),
            "effect_pct": c["Delta_pct"], "effect_z": c["Delta_z"], "p_exact": c["p_exact"], "q_bh": c["q_value"],
            "A": a["Amplification_contrast"], "A_ci_lo": a["CI_lo"], "A_ci_hi": a["CI_hi"],
            "A_p": a["p_value"], "A_q": a["q_value"], "SE_A": a["SE"], "MDE80": MDE80_FACTOR * a["SE"],
        })
        shifts.append({"baseline": baseline, "smoothing": int(smoothing), "family": family, "outcome": col,
                       "S_effect": int(c["N_shifts"]), "S_A": int(a["N_shifts"])})
    return pd.DataFrame(rows, columns=list(COLUMNS)), pd.DataFrame(shifts)


def shift_summary(shifts: pd.DataFrame) -> pd.DataFrame:
    """Per baseline (and smoothing and family): range of shift counts S and the exact-p floor 1/(S+1)."""
    keys = ["baseline", "smoothing", "family"]
    s = shifts.melt(id_vars=keys, value_vars=["S_effect", "S_A"], value_name="S")
    s = s[s["S"] > 0]  # 0 = estimate undefined, so no randomization was run
    out = s.groupby(keys, sort=False)["S"].agg(S_min="min", S_max="max")
    out["p_floor_at_S_min"] = 1.0 / (out["S_min"] + 1)
    out["p_floor_at_S_max"] = 1.0 / (out["S_max"] + 1)
    return out.reset_index()
