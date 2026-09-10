import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import tempfile
from unittest.mock import patch

import componergy.indices.pipeline as pipeline_mod
from componergy.indices.pipeline import build_all_indices


def _make_synthetic_monthly_climate(n_years=15, seed=0):
    rng = np.random.default_rng(seed)
    lats = np.array([33.0, 36.0])
    lons = np.array([-122.0, -120.0])
    times = pd.date_range("2000-01-01", periods=12 * n_years, freq="MS")
    n_time = len(times)
    months = times.month.values

    tavg = (15 + 10 * np.sin(2 * np.pi * (months - 4) / 12))[:, None, None] + rng.normal(0, 3, (n_time, 2, 2))
    tmax = tavg + rng.uniform(5, 12, (n_time, 2, 2))
    tmin = tavg - rng.uniform(5, 12, (n_time, 2, 2))
    prcp = np.clip((40 - 30 * np.sin(2 * np.pi * (months - 4) / 12))[:, None, None] + rng.normal(0, 15, (n_time, 2, 2)),
                   0, None)

    return xr.Dataset(
        {
            "tavg": (("time", "lat", "lon"), tavg), "tmax": (("time", "lat", "lon"), tmax),
            "tmin": (("time", "lat", "lon"), tmin), "prcp": (("time", "lat", "lon"), prcp),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )


def test_build_all_indices_produces_every_expected_variable():
    ds = _make_synthetic_monthly_climate()
    result = build_all_indices(ds, n_jobs=1)

    expected = {"pet", "sti", "sti_climatology_mean", "sti_climatology_std", "sti_climatology_n"}
    expected |= {f"sapei_{n}m" for n in [3, 6, 9, 12]}
    expected |= {f"scdhi_{n}m" for n in [3, 6, 9, 12]}
    assert expected.issubset(set(result.data_vars))


def test_pet_is_non_negative_everywhere_defined():
    ds = _make_synthetic_monthly_climate()
    result = build_all_indices(ds, n_jobs=1)
    pet_vals = result["pet"].values
    assert (pet_vals[~np.isnan(pet_vals)] >= 0).all()


def test_main_writes_indices_file_and_is_idempotent():
    monthly_ds = _make_synthetic_monthly_climate()
    test_dir = Path(tempfile.mkdtemp())
    test_monthly_file = test_dir / "monthly.nc"
    test_indices_file = test_dir / "indices.nc"
    monthly_ds.to_netcdf(test_monthly_file)

    with patch.object(pipeline_mod, "NOAA_MONTHLY_FILE", test_monthly_file), \
            patch.object(pipeline_mod, "INDICES_FILE", test_indices_file), \
            patch.object(pipeline_mod, "ensure_dirs", return_value=None):
        pipeline_mod.main(n_jobs=1)
        assert test_indices_file.exists()
        result = xr.open_dataset(test_indices_file)
        assert {"pet", "sti", "sapei_3m", "scdhi_3m"}.issubset(result.data_vars)

        mtime_before = test_indices_file.stat().st_mtime
        pipeline_mod.main(n_jobs=1)
        assert test_indices_file.stat().st_mtime == mtime_before
