import numpy as np
import pandas as pd

from componergy.analysis.demand_crosscheck import compare_demand_sources
from componergy.preprocessing.consumption_cec import statewide_monthly_mwh


def _seasonal(n=204, seed=0, noise=0.02):
    idx = pd.date_range("2008-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    season = 1.0 + 0.25 * np.sin(2 * np.pi * (idx.month.values - 4) / 12)
    return pd.Series(20000 * season * (1 + rng.normal(0, noise, n)), index=idx)
