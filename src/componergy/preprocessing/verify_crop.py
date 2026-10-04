"""Scan cropped NOAA files for corruption (truncated/incomplete writes)."""

from __future__ import annotations

import xarray as xr

EXPECTED_VARS = ("tavg", "tmax", "tmin", "prcp")


def check_file(nc_file) -> str | None:
    """Try to open and fully read a cropped file. Returns None if OK,
    else a short error description."""
    try:
        ds = xr.open_dataset(nc_file)
        for var in EXPECTED_VARS:
            _ = ds[var].values
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
