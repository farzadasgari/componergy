"""Relative importance of correlated predictors: LMG ("general dominance") decomposition."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from componergy.analysis.block_bootstrap import moving_block_indices
from componergy.analysis.lag_scan import block_permute


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    if X.shape[1] == 0:
        return 0.0
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(resid @ resid) / sst if sst > 0 else 0.0
