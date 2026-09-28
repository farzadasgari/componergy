"""
Standardized Compound Drought and Heat Index (SCDHI), monthly.
"""

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
    """
    Pool (SAPEI, STI) pairs across a sample of valid cells and rank
    copula families once for the whole domain.

    Parameters
    ----------
    sapei_grid, sti_grid : np.ndarray
        (time, lat, lon) arrays of standardized SAPEI (one timescale) and STI.
    seed : int
        Random seed for cell sampling, for reproducibility.

    Returns
    -------
    list[dict]
        compare_families() output -- ranked by AIC, with BIC and KS
        p-value also reported so the selection stays inspectable.
    """
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


def compute_scdhi_for_cell(sapei_by_timescale: dict, sti: np.ndarray, family_name: str) -> dict:
    """
    Fit ONE fixed copula family's parameters and compute SCDHI, for ONE grid cell.

    Handles all requested SAPEI timescales for that cell in a single call,
    to keep per-task overhead low when parallelized across cells.

    Parameters
    ----------
    sapei_by_timescale : dict[int, np.ndarray]
        {n_months: standardized SAPEI series} for one cell.
    sti : np.ndarray
        Standardized STI series for the same cell.
    family_name : str
        Which entry of FAMILIES to fit (chosen once, domain-wide -- see
        select_primary_copula_family).

    Returns
    -------
    dict[int, np.ndarray]
        {n_months: scdhi series}, NaN wherever fewer than
        MIN_SAMPLES_FOR_COPULA valid (SAPEI, STI) pairs are available.
    """
    fam = FAMILIES[family_name]
    result = {}
    for n_months, sapei in sapei_by_timescale.items():
        valid = ~(np.isnan(sapei) | np.isnan(sti))
        scdhi = np.full_like(sapei, np.nan)
        if valid.sum() < MIN_SAMPLES_FOR_COPULA:
            result[n_months] = scdhi
            continue

        u = np.clip(norm.cdf(sapei[valid]), 1e-6, 1 - 1e-6)
        v = np.clip(norm.cdf(sti[valid]), 1e-6, 1 - 1e-6)

        params = fam["fit"](u, v)
        c = fam["cdf"](u, v, params)
        p = np.clip(u - c, 1e-10, 1 - 1e-10)

        standardized = gringorten_standardize(p)
        scdhi[valid] = standardized
        result[n_months] = scdhi
    return result


def add_scdhi(monthly_ds: xr.Dataset, family_name: str = None, n_jobs: int = -1) -> xr.Dataset:
    """
    Add scdhi_3m/6m/9m/12m to a monthly climate dataset.

    Requires sapei_3m/6m/9m/12m and sti to already be present (see
    sapei.add_sapei, sti.add_sti).

    Parameters
    ----------
    monthly_ds : xr.Dataset
        Must contain sti and sapei_{3,6,9,12}m.
    family_name : str, optional
        Fix the copula family instead of auto-selecting it (useful to
        reuse a family already chosen on a prior run, or for testing).
    n_jobs : int
        Passed to joblib.Parallel; -1 uses all available cores.

    Returns
    -------
    xr.Dataset
        `monthly_ds` with scdhi_3m, scdhi_6m, scdhi_9m, scdhi_12m added,
        and the chosen family recorded in .attrs["scdhi_copula_family"].
    """
    sapei_timescales = [3, 6, 9, 12]
    sapei_grids = {n: monthly_ds[f"sapei_{n}m"].values for n in sapei_timescales}
    sti_grid = monthly_ds["sti"].values

    if family_name is None:
        ranking = select_primary_copula_family(sapei_grids[3], sti_grid)
        family_name = ranking[0]["family"]
        print(f"selected primary copula family: {family_name}")
        for r in ranking:
            print(f"  {r['family']:10s} AIC={r['aic']:9.2f}  BIC={r['bic']:9.2f}  KS_pval={r['ks_pval']}")

    lats = monthly_ds["lat"].values
    lons = monthly_ds["lon"].values
    n_time = monthly_ds.sizes["time"]

    valid_cell = ~np.isnan(sti_grid).all(axis=0)
    cell_indices = [(i, j) for i in range(len(lats)) for j in range(len(lons)) if valid_cell[i, j]]

    cell_inputs = [
        ({n: sapei_grids[n][:, i, j] for n in sapei_timescales}, sti_grid[:, i, j])
        for i, j in cell_indices
    ]

    cell_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_scdhi_for_cell)(sapei_by_timescale, sti_cell, family_name)
        for sapei_by_timescale, sti_cell in tqdm(cell_inputs, desc="fitting SCDHI per cell", unit="cell")
    )

    scdhi_arrays = {n: np.full((n_time, len(lats), len(lons)), np.nan) for n in sapei_timescales}
    for (i_lat, i_lon), cell_result in zip(cell_indices, cell_results):
        for n in sapei_timescales:
            scdhi_arrays[n][:, i_lat, i_lon] = cell_result[n]

    out = monthly_ds.copy()
    for n in sapei_timescales:
        out[f"scdhi_{n}m"] = (("time", "lat", "lon"), scdhi_arrays[n])
    out.attrs["scdhi_copula_family"] = family_name
    return out
