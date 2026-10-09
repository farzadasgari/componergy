"""Run manifest: what exactly produced a set of results."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = ("numpy", "pandas", "xarray", "scipy", "matplotlib", "geopandas", "regionmask", "netCDF4",
            "rioxarray", "dask", "statsmodels", "joblib", "cartopy", "shapely", "pyproj")
SEEDS = {"bootstrap_raf": 42, "superadditivity": 42, "lag_scan_randomization": 42,
         "relative_importance_bootstrap": 42, "return_period_bootstrap": 42, "copula_selection": 0}
HASH_LIMIT_BYTES = 500 * 1024 * 1024


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
