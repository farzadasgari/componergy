"""
Crash-safe NetCDF writing.

Writing directly to the final path means a crash mid-write (OOM-kill,
power loss, etc.) leaves a corrupted, incomplete file sitting at exactly
the path every idempotency check in this project uses to decide "is this
already done" -- so a corrupted file gets silently treated as finished
and never regenerated. atomic_to_netcdf() writes to a sibling .tmp file
first and only renames it into place once the write fully succeeds, so
the final path never exists in a half-written state.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr


def atomic_to_netcdf(ds: xr.Dataset, path, encoding: dict = None) -> None:
    """
    Write `ds` to `path` atomically.

    On POSIX filesystems, Path.replace() is an atomic rename as long as
    source and destination are on the same filesystem (true here, since
    the .tmp file is a sibling of the final path).
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    ds.to_netcdf(tmp_path, encoding=encoding)
    tmp_path.replace(path)


def atomic_to_csv(df, path, **kwargs) -> None:
    """Write `df` to `path` as CSV atomically, same rationale as atomic_to_netcdf."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, **kwargs)
    tmp_path.replace(path)
