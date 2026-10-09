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
