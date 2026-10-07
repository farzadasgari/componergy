import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import tempfile
from unittest.mock import patch

import componergy.preprocessing.aggregate_monthly as agg_mod


def _make_daily_file(directory, year, month, n_days, tavg_val, tmax_val, tmin_val, prcp_val, missing_days=0, lats=None,
                     lons=None):
    lats = lats if lats is not None else np.array([35.0, 36.0])
    lons = lons if lons is not None else np.array([-120.0, -119.0])
    dates = pd.date_range(f"{year}-{month:02d}-01", periods=n_days, freq="D")
    shape = (n_days, len(lats), len(lons))
    tavg = np.full(shape, tavg_val, dtype="float32")
    tmax = np.full(shape, tmax_val, dtype="float32")
    tmin = np.full(shape, tmin_val, dtype="float32")
    prcp = np.full(shape, prcp_val, dtype="float32")
    if missing_days:
        tavg[:missing_days] = np.nan
        tmax[:missing_days] = np.nan
        tmin[:missing_days] = np.nan
        prcp[:missing_days] = np.nan
    ds = xr.Dataset(
        {"tavg": (("time", "lat", "lon"), tavg), "tmax": (("time", "lat", "lon"), tmax),
         "tmin": (("time", "lat", "lon"), tmin), "prcp": (("time", "lat", "lon"), prcp)},
        coords={"time": dates, "lat": lats, "lon": lons},
    )
    path = directory / f"ca-ncdd-{year}{month:02d}-grd-scaled.nc"
    ds.to_netcdf(path)
    return path


def test_complete_month_aggregates_correctly():
    test_dir = Path(tempfile.mkdtemp())
    _make_daily_file(test_dir, 2020, 1, 31, tavg_val=10.0, tmax_val=15.0, tmin_val=5.0, prcp_val=2.0)

    result = agg_mod.aggregate_file(test_dir / "ca-ncdd-202001-grd-scaled.nc")
    assert abs(float(result["tavg"].isel(time=0, lat=0, lon=0).values) - 10.0) < 1e-5
    assert abs(float(result["prcp"].isel(time=0, lat=0, lon=0).values) - 62.0) < 1e-4  # 31 days * 2mm


def test_minor_gap_still_aggregates_from_valid_days():
    test_dir = Path(tempfile.mkdtemp())
    _make_daily_file(test_dir, 2020, 2, 29, tavg_val=12.0, tmax_val=17.0, tmin_val=7.0, prcp_val=1.0, missing_days=2)

    result = agg_mod.aggregate_file(test_dir / "ca-ncdd-202002-grd-scaled.nc")
    assert abs(float(result["tavg"].isel(time=0, lat=0, lon=0).values) - 12.0) < 1e-5
    assert abs(float(result["prcp"].isel(time=0, lat=0, lon=0).values) - 27.0) < 1e-4  # 27 valid days * 1mm


def test_major_gap_is_nan_rather_than_partial_total():
    test_dir = Path(tempfile.mkdtemp())
    _make_daily_file(test_dir, 2020, 3, 31, tavg_val=14.0, tmax_val=19.0, tmin_val=9.0, prcp_val=3.0, missing_days=20)

    result = agg_mod.aggregate_file(test_dir / "ca-ncdd-202003-grd-scaled.nc")
    assert np.isnan(float(result["tavg"].isel(time=0, lat=0, lon=0).values))
    assert np.isnan(float(result["prcp"].isel(time=0, lat=0, lon=0).values))


def test_crash_mid_write_never_leaves_a_corrupted_final_file():
    test_dir = Path(tempfile.mkdtemp())
    _make_daily_file(test_dir, 2020, 5, 31, tavg_val=20.0, tmax_val=25.0, tmin_val=15.0, prcp_val=0.5)

    test_monthly_file = test_dir / "monthly_output.nc"

    class SimulatedCrash(Exception):
        pass

    with patch.object(agg_mod, "NOAA_CA_DIR", test_dir), \
            patch.object(agg_mod, "NOAA_MONTHLY_FILE", test_monthly_file), \
            patch.object(agg_mod, "ensure_dirs", return_value=None), \
            patch("xarray.Dataset.to_netcdf", side_effect=SimulatedCrash("simulated oom-kill")):
        try:
            agg_mod.main()
        except SimulatedCrash:
            pass

    assert not test_monthly_file.exists(), (
        "final path must not exist after a crash mid-write -- "
        "otherwise the idempotency check will skip regenerating it forever"
    )
