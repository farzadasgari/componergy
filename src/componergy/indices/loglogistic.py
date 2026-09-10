from __future__ import annotations

import numpy as np
from scipy.special import gamma as gamma_func


def fit_loglogistic_lmoments(x: np.ndarray) -> tuple[float, float, float]:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 10:
        raise ValueError(f"Too few samples ({n}) to fit a log-logistic distribution.")
    i = np.arange(1, n + 1)

    w0 = np.mean(x)
    w1 = np.mean(((n - i) / (n - 1)) * x)
    w2 = np.mean(((n - i) * (n - i - 1)) / ((n - 1) * (n - 2)) * x)

    beta = (2 * w1 - w0) / (6 * w1 - w0 - 6 * w2)
    g1 = gamma_func(1 + 1 / beta)
    g2 = gamma_func(1 - 1 / beta)
    alpha = (w0 - 2 * w1) * beta / (g1 * g2)
    gamma_param = w0 - alpha * g1 * g2

    return alpha, beta, gamma_param


def loglogistic_cdf(x: np.ndarray, alpha: float, beta: float, gamma_param: float) -> np.ndarray:
    return 1.0 / (1.0 + (alpha / (x - gamma_param)) ** beta)
