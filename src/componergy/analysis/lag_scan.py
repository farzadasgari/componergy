"""Best-lag regression with inference that accounts for the lag search."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_LAGS = tuple(range(-6, 7))
DEFAULT_BLOCK = 24
DEFAULT_N_PERM = 1999


def _lagged(x: np.ndarray, y: np.ndarray, lag: int):
    """Same pairing convention as composite_stats.best_lag_slope."""
    if lag < 0:
        xl = x[-lag:]
        yl = y[:len(xl)]
    elif lag > 0:
        xl, yl = x[:-lag], y[lag:]
    else:
        xl, yl = x, y
    return xl, yl
