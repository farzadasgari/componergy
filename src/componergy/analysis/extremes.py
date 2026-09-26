"""
Extreme-value analysis of the systemic risk series: empirical plotting
positions plus two fitted extreme-value models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MONTHS_PER_YEAR = 12


def empirical_return_period(series: pd.Series) -> pd.DataFrame:
    """Weibull plotting position (n+1)/rank, converted to years."""
    s = series.dropna().sort_values(ascending=False)
    n = len(s)
    ranks = np.arange(1, n + 1)
    return pd.DataFrame({
        "Date": s.index,
        "Value": s.values,
        "Return_Period_Years": ((n + 1.0) / ranks) / MONTHS_PER_YEAR,
    })
