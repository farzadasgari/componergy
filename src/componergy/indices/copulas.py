from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau, kstest, multivariate_normal, norm

EPS = 1e-6