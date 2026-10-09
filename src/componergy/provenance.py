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


def directory_fingerprint(path, pattern: str = "*") -> dict:
    files = sorted(p for p in Path(path).rglob(pattern) if p.is_file())
    listing = "\n".join(f"{p.relative_to(path)}:{p.stat().st_size}" for p in files)
    return {"n_files": len(files), "total_bytes": int(sum(p.stat().st_size for p in files)),
            "listing_sha256": hashlib.sha256(listing.encode()).hexdigest()}
