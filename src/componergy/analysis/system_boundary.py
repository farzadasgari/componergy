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
