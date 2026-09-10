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


def gaussian_cdf(u, v, params):
    rho = params["rho"]
    u, v = _clip(u, v)
    x, y = norm.ppf(u), norm.ppf(v)
    mvn = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    pts = np.column_stack([x, y])
    return np.array([mvn.cdf(p) for p in pts])


def gaussian_loglik(u, v, params):
    rho = params["rho"]
    u, v = _clip(u, v)
    x, y = norm.ppf(u), norm.ppf(v)
    density = (1 / np.sqrt(1 - rho ** 2)) * np.exp(
        -(rho ** 2 * (x ** 2 + y ** 2) - 2 * rho * x * y) / (2 * (1 - rho ** 2))
    )
    return float(np.sum(np.log(np.clip(density, 1e-300, None))))


def gaussian_fit(u, v):
    u, v = _clip(u, v)
    x, y = norm.ppf(u), norm.ppf(v)
    rho = float(np.corrcoef(x, y)[0, 1])
    rho = float(np.clip(rho, -0.999, 0.999))
    return {"rho": rho}


GAUSSIAN = {"fit": gaussian_fit, "loglik": gaussian_loglik, "cdf": gaussian_cdf, "n_params": 1}


def _clayton_cdf_base(u, v, theta):
    return np.maximum(u ** (-theta) + v ** (-theta) - 1, 1e-12) ** (-1 / theta)


def _clayton_density_base(u, v, theta):
    return (
            (theta + 1)
            * (u * v) ** (-theta - 1)
            * (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta - 2)
    )


def _clayton_tau_to_theta(tau):
    return 2 * tau / max(1 - tau, 1e-6)


def _gumbel_cdf_base(u, v, theta):
    A, B = -np.log(u), -np.log(v)
    w = A ** theta + B ** theta
    return np.exp(-(w ** (1 / theta)))
