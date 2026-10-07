import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import multivariate_normal as mvn

from componergy.indices.scdhi import add_scdhi


def _make_test_dataset(true_rho=-0.5, n_years=25, seed=3, mask_one_cell=True):
    rng = np.random.default_rng(seed)
    lats = np.array([33.0, 36.0, 39.0])
    lons = np.array([-122.0, -120.0, -118.0])
    times = pd.date_range("2000-01-01", periods=12 * n_years, freq="MS")
    n_time = len(times)

    sapei_grids = {n: np.full((n_time, 3, 3), np.nan) for n in [3, 6, 9, 12]}
    sti_grid = np.full((n_time, 3, 3), np.nan)

    for i in range(3):
        for j in range(3):
            xy = mvn.rvs(mean=[0, 0], cov=[[1, true_rho], [true_rho, 1]], size=n_time, random_state=rng)
            sti_grid[:, i, j] = xy[:, 1]
            for n in [3, 6, 9, 12]:
                sapei_grids[n][:, i, j] = xy[:, 0]

    if mask_one_cell:
        sti_grid[:, 2, 2] = np.nan
        for n in [3, 6, 9, 12]:
            sapei_grids[n][:, 2, 2] = np.nan

    data_vars = {"sti": (("time", "lat", "lon"), sti_grid)}
    for n in [3, 6, 9, 12]:
        data_vars[f"sapei_{n}m"] = (("time", "lat", "lon"), sapei_grids[n])
    return xr.Dataset(data_vars, coords={"time": times, "lat": lats, "lon": lons}), sapei_grids, sti_grid


def test_family_selection_identifies_true_generating_family():
    ds, _, _ = _make_test_dataset()
    result = add_scdhi(ds, n_jobs=1)
    assert result.attrs["scdhi_copula_family"] == "gaussian"


def test_parallel_matches_sequential():
    ds, _, _ = _make_test_dataset()
    result_seq = add_scdhi(ds, n_jobs=1)
    result_par = add_scdhi(ds, n_jobs=2, family_name=result_seq.attrs["scdhi_copula_family"])

    for n in [3, 6, 9, 12]:
        a, b = result_seq[f"scdhi_{n}m"].values, result_par[f"scdhi_{n}m"].values
        both_nan = np.isnan(a) & np.isnan(b)
        assert (np.isclose(a, b, equal_nan=False) | both_nan).all(), f"scdhi_{n}m differs"


def test_fully_masked_cell_stays_nan_others_dont():
    ds, _, _ = _make_test_dataset(mask_one_cell=True)
    result = add_scdhi(ds, n_jobs=1)
    assert np.isnan(result["scdhi_3m"].values[:, 2, 2]).all()
    assert not np.isnan(result["scdhi_3m"].values[:, 0, 0]).all()


def test_compound_extreme_months_score_lower_than_normal_months():
    ds, sapei_grids, sti_grid = _make_test_dataset()
    result = add_scdhi(ds, n_jobs=1)

    cell_sapei, cell_sti = sapei_grids[3][:, 0, 0], sti_grid[:, 0, 0]
    cell_scdhi = result["scdhi_3m"].values[:, 0, 0]

    compound_mask = (cell_sapei < np.nanpercentile(cell_sapei, 10)) & (cell_sti > np.nanpercentile(cell_sti, 90))
    normal_mask = (np.abs(cell_sapei) < 0.3) & (np.abs(cell_sti) < 0.3)

    mean_compound = np.nanmean(cell_scdhi[compound_mask])
    mean_normal = np.nanmean(cell_scdhi[normal_mask])
    assert mean_compound < mean_normal
    assert mean_compound < -1.0
