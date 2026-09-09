from __future__ import annotations

from pathlib import Path

import xarray as xr


def atomic_to_netcdf(ds: xr.Dataset, path, encoding: dict = None) -> None:
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    ds.to_netcdf(tmp_path, encoding=encoding)
    tmp_path.replace(path)
