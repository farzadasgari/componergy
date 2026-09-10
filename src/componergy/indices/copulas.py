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


def frank_cdf(u, v, params):
    theta = params["theta"]
    u, v = _clip(u, v)
    if abs(theta) < 1e-6:
        return u * v
    return -1 / theta * np.log(
        1 + (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) / (np.exp(-theta) - 1)
    )


def frank_loglik(u, v, params):
    u, v = _clip(u, v)
    d = np.clip(_frank_density(u, v, params["theta"]), 1e-300, None)
    return float(np.sum(np.log(d)))


def frank_fit(u, v):
    u, v = _clip(u, v)

    def neg_ll(theta):
        return -np.sum(np.log(np.clip(_frank_density(u, v, theta), 1e-300, None)))

    result = minimize_scalar(neg_ll, bounds=(-30, 30), method="bounded")
    return {"theta": float(result.x)}


FRANK = {"fit": frank_fit, "loglik": frank_loglik, "cdf": frank_cdf, "n_params": 1}
