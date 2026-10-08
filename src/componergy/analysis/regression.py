"""
Event effects with seasonal and trend controls, and inference that stays valid
when event months come in a few long episodes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TREND_SPECS = ("none", "linear", "quadratic", "year_fe")
DEFAULT_HAC_LAGS = 12
DEFAULT_MIN_EVENT = 3
DEFAULT_MIN_SHIFT = 12
CI_LEVEL = 0.95
_Z = float(stats.norm.ppf(0.5 + CI_LEVEL / 2))


def year_fraction(index) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    return np.asarray(idx.year + (idx.month - 0.5) / 12.0, dtype=float)
