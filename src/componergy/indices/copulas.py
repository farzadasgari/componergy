from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau, kstest, multivariate_normal, norm

EPS = 1e-6


def _clip(u, v):
    return np.clip(u, EPS, 1 - EPS), np.clip(v, EPS, 1 - EPS)
