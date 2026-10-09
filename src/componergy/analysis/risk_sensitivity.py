"""How much of the systemic-risk ranking is the index, and how much is the data?"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

REFERENCE = "product (manuscript)"


def build_risk_variants(gen_stress: pd.Series, demand_z: pd.Series,
                        generation_z: pd.Series | None = None) -> pd.DataFrame:
    """One column per risk-index definition over the months all inputs share."""
    common = gen_stress.dropna().index.intersection(demand_z.dropna().index)
    if generation_z is not None:
        common = common.intersection(generation_z.dropna().index)
    g, d = gen_stress.loc[common].astype(float), demand_z.loc[common].astype(float)

    out = {
        REFERENCE: g * d,
        "product of positive parts (stress only)": g.clip(lower=0) * d.clip(lower=0),
        "additive (z-sum)": g + d,
    }
    if generation_z is not None:
        out["observed generation shortfall x demand"] = (-generation_z.loc[common].astype(float)) * d
    return pd.DataFrame(out)


def double_calm_share(gen_stress: pd.Series, demand_z: pd.Series) -> dict:
    """Of the months the product scores as positive risk, how many have BOTH factors negative?"""
    common = gen_stress.dropna().index.intersection(demand_z.dropna().index)
    g, d = gen_stress.loc[common], demand_z.loc[common]
    prod = g * d
    positive = prod > 0
    both_negative = positive & (g < 0) & (d < 0)
    n_pos = int(positive.sum())
    return {"n_months": int(len(common)), "n_positive_risk_months": n_pos,
            "n_positive_from_both_negative": int(both_negative.sum()),
            "share_of_positive_months_that_are_double_calm":
                float(both_negative.sum() / n_pos) if n_pos else np.nan}


def compare_risk_variants(variants: pd.DataFrame, reference: str = REFERENCE, top_k=(5, 10)) -> pd.DataFrame:
    """Each variant against the reference: rank agreement, overlap of the worst months, and the peak month."""
    ref = variants[reference]
    ref_peak = ref.idxmax()
    rows = []
    for name in variants.columns:
        v = variants[name]
        rho = float(stats.spearmanr(ref, v)[0])
        tau = float(stats.kendalltau(ref, v)[0])
        ranks = v.rank(ascending=False, method="min")
        row = {"Variant": name, "Spearman_vs_reference": rho, "Kendall_vs_reference": tau,
               "Peak_month": str(v.idxmax().date()),
               "Same_peak_month_as_reference": bool(v.idxmax() == ref_peak),
               "Rank_of_reference_peak_month": int(ranks.loc[ref_peak])}
        for k in top_k:
            row[f"Top{k}_overlap_with_reference"] = int(len(set(ref.nlargest(k).index) & set(v.nlargest(k).index)))
        rows.append(row)
    return pd.DataFrame(rows)
