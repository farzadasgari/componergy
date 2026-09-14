import numpy as np
import pandas as pd
import xarray as xr

from componergy.analysis.events import (
    compute_percentile_threshold,
    classify_heatwave,
    classify_drought,
    classify_events,
    normal_mask_exclude_all_events,
    normal_mask_opposite_extreme,
    compute_delta_z,
)


def _make_test_arrays(seed=5, n_time=900):
    rng = np.random.default_rng(seed)
    lats, lons = np.array([35.0, 38.0]), np.array([-120.0, -119.0])
    times = pd.date_range("1951-01-01", periods=n_time, freq="MS")
    coords = {"time": times, "lat": lats, "lon": lons}

    sti = xr.DataArray(rng.normal(0, 1, (n_time, 2, 2)), dims=("time", "lat", "lon"), coords=coords)
    sapei = xr.DataArray(rng.normal(0, 1, (n_time, 2, 2)), dims=("time", "lat", "lon"), coords=coords)
    scdhi = xr.DataArray(rng.normal(0, 1, (n_time, 2, 2)), dims=("time", "lat", "lon"), coords=coords)
    return sti, sapei, scdhi


def test_percentile_threshold_matches_manual_numpy_calc():
    sti, _, _ = _make_test_arrays()
    thr = compute_percentile_threshold(sti, 0.90)
    for i in range(2):
        for j in range(2):
            expected = np.percentile(sti.values[:, i, j], 90)
            assert abs(expected - float(thr.isel(lat=i, lon=j).values)) < 1e-9


def test_classify_heatwave_and_drought_flag_approximately_ten_percent():
    sti, sapei, _ = _make_test_arrays()
    heat = classify_heatwave(sti, 0.90)
    drought = classify_drought(sapei, 0.10)
    assert abs(float(heat.isel(lat=0, lon=0).mean()) - 0.10) < 0.02
    assert abs(float(drought.isel(lat=0, lon=0).mean()) - 0.10) < 0.02


def test_classify_events_dataset_combines_without_coordinate_conflict():
    sti, sapei, scdhi = _make_test_arrays()
    result = classify_events(sti, sapei, scdhi)
    assert set(result.data_vars) == {
        "is_heatwave", "is_drought", "is_compound", "is_heatwave_only", "is_drought_only",
    }


def test_heatwave_only_and_drought_only_never_overlap_each_other():
    sti, sapei, scdhi = _make_test_arrays()
    result = classify_events(sti, sapei, scdhi)

    raw_overlap = result["is_heatwave"] & result["is_drought"]
    assert bool(raw_overlap.any()), "sanity: random data should have some raw overlap to test against"

    assert not bool((result["is_heatwave_only"] & result["is_drought"]).any())
    assert not bool((result["is_drought_only"] & result["is_heatwave"]).any())


def test_delta_z_matches_manual_mean_difference():
    sti, sapei, scdhi = _make_test_arrays()
    result = classify_events(sti, sapei, scdhi)
    normal = normal_mask_exclude_all_events(result["is_heatwave"], result["is_drought"], result["is_compound"])
    dz = compute_delta_z(sti, result["is_heatwave"], normal)

    cell_sti = sti.isel(lat=0, lon=0).values
    cell_heat = result["is_heatwave"].isel(lat=0, lon=0).values
    cell_normal = normal.isel(lat=0, lon=0).values
    expected = cell_sti[cell_heat].mean() - cell_sti[cell_normal].mean()

    assert abs(expected - float(dz.isel(lat=0, lon=0).values)) < 1e-9


def test_normal_mask_opposite_extreme_differs_from_exclude_all_events():
    sti, sapei, scdhi = _make_test_arrays()
    result = classify_events(sti, sapei, scdhi)
    normal_exclude = normal_mask_exclude_all_events(result["is_heatwave"], result["is_drought"], result["is_compound"])
    normal_opposite = normal_mask_opposite_extreme(scdhi, 0.90)

    assert not bool((normal_exclude == normal_opposite).all())
