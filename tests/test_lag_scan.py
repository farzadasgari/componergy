import numpy as np
import pandas as pd
import pytest

from componergy.analysis.lag_scan import best_lag_stats, best_lag_randomization, block_permute
from componergy.figures.composite_stats import best_lag_slope


def _ar1(n, phi, rng):
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + rng.normal()
    return e
