from pathlib import Path
import tempfile
from unittest.mock import patch

import componergy.paths as paths


def test_pipeline_dirs_contains_no_file_path_constants():
    file_constants = {
        paths.NOAA_MONTHLY_FILE,
        paths.INDICES_FILE,
        paths.EIA_GENERATION_RAW_FILE,
        paths.GENERATION_MONTHLY_FILE,
        paths.MANIFEST_PATH,
    }
    for d in paths._PIPELINE_DIRS:
        assert d not in file_constants, f"{d} is a file path and must not be in _PIPELINE_DIRS"


def test_ensure_dirs_does_not_crash_when_file_exists_in_processed_dir():
    """End-to-end reproduction of the actual production failure: a real
    file already sitting in PROCESSED_DATA_DIR must not stop ensure_dirs()
    from creating the other pipeline directories.
    """
    test_root = Path(tempfile.mkdtemp())
    fake_processed = test_root / "processed"
    fake_processed.mkdir()
    (fake_processed / "california_monthly_climate.nc").write_text("pretend netcdf content")

    fake_dirs = (
        test_root / "raw",
        test_root / "intermediate",
        fake_processed,
        test_root / "raw" / "climate",
        test_root / "raw" / "boundaries",
        test_root / "intermediate" / "nclimgrid-ca",
        test_root / "figures",
    )

    with patch.object(paths, "_PIPELINE_DIRS", fake_dirs):
        paths.ensure_dirs()

    for d in fake_dirs:
        assert d.is_dir()
    assert (fake_processed / "california_monthly_climate.nc").exists()


def test_ensure_dirs_creates_all_directories():
    test_root = Path(tempfile.mkdtemp())
    fake_dirs = tuple(test_root / name for name in ["a", "b", "c"])

    with patch.object(paths, "_PIPELINE_DIRS", fake_dirs):
        paths.ensure_dirs()

    for d in fake_dirs:
        assert d.is_dir()
