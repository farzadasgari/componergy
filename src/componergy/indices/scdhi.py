from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm import tqdm

from componergy.indices.copulas import FAMILIES, compare_families
from componergy.indices.gringorten import gringorten_standardize

MIN_SAMPLES_FOR_COPULA = 30
POOLING_SAMPLE_CELLS = 200


def select_primary_copula_family(sapei_grid: np.ndarray, sti_grid: np.ndarray, seed: int = 0) -> list:
    n_time, n_lat, n_lon = sapei_grid.shape
    valid_cells = [(i, j) for i in range(n_lat) for j in range(n_lon)
                   if (~np.isnan(sapei_grid[:, i, j]) & ~np.isnan(sti_grid[:, i, j])).sum() >= MIN_SAMPLES_FOR_COPULA]

    rng = np.random.default_rng(seed)
    sample_cells = valid_cells if len(valid_cells) <= POOLING_SAMPLE_CELLS else \
        [valid_cells[i] for i in rng.choice(len(valid_cells), POOLING_SAMPLE_CELLS, replace=False)]

    pooled_u, pooled_v = [], []
    for i, j in sample_cells:
        sapei_cell, sti_cell = sapei_grid[:, i, j], sti_grid[:, i, j]
        valid = ~(np.isnan(sapei_cell) | np.isnan(sti_cell))
        pooled_u.append(norm.cdf(sapei_cell[valid]))
        pooled_v.append(norm.cdf(sti_cell[valid]))

    u, v = np.concatenate(pooled_u), np.concatenate(pooled_v)
    return compare_families(u, v)
