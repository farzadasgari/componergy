from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau, kstest, multivariate_normal, norm

EPS = 1e-6


def _clip(u, v):
    return np.clip(u, EPS, 1 - EPS), np.clip(v, EPS, 1 - EPS)


def _frank_density(u, v, theta):
    if abs(theta) < 1e-6:
        return np.ones_like(u)
    num = theta * (1 - np.exp(-theta)) * np.exp(-theta * (u + v))
    denom = ((1 - np.exp(-theta)) - (1 - np.exp(-theta * u)) * (1 - np.exp(-theta * v))) ** 2
    return num / denom
