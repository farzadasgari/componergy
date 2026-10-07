import numpy as np
import xarray as xr
from pathlib import Path
import tempfile

from componergy.preprocessing.verify_crop import check_file, find_corrupted_files


def _make_good_file(directory, name):
    ds = xr.Dataset({v: (("time", "lat", "lon"), np.random.rand(30, 5, 5).astype("float32"))
                     for v in ["tavg", "tmax", "tmin", "prcp"]})
    path = directory / name
    ds.to_netcdf(path)
    return path


def test_good_file_passes_check():
    test_dir = Path(tempfile.mkdtemp())
    path = _make_good_file(test_dir, "ca-ncdd-202001-grd-scaled.nc")
    assert check_file(path) is None


def test_truncated_file_is_detected():
    test_dir = Path(tempfile.mkdtemp())
    path = _make_good_file(test_dir, "ca-ncdd-202003-grd-scaled.nc")
    with open(path, "r+b") as f:
        f.truncate(path.stat().st_size // 10)
    assert check_file(path) is not None


def test_find_corrupted_files_identifies_only_the_bad_one():
    test_dir = Path(tempfile.mkdtemp())
    _make_good_file(test_dir, "ca-ncdd-202001-grd-scaled.nc")
    _make_good_file(test_dir, "ca-ncdd-202002-grd-scaled.nc")
    bad_path = _make_good_file(test_dir, "ca-ncdd-202003-grd-scaled.nc")
    with open(bad_path, "r+b") as f:
        f.truncate(bad_path.stat().st_size // 10)

    corrupted = find_corrupted_files(test_dir)
    assert len(corrupted) == 1
    assert corrupted[0][0].name == "ca-ncdd-202003-grd-scaled.nc"
