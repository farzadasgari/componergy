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


def find_corrupted_files(directory) -> list[tuple]:
    """Check every .nc file in `directory`, return [(path, error), ...] for
    any that fail to open or are missing an expected variable."""
    results = []
    for nc_file in sorted(directory.glob("*.nc")):
        error = check_file(nc_file)
        if error is not None:
            results.append((nc_file, error))
    return results


def main(delete: bool = False) -> None:
    from componergy.paths import NOAA_CA_DIR

    print(f"scanning {NOAA_CA_DIR} ...")
    corrupted = find_corrupted_files(NOAA_CA_DIR)

    if not corrupted:
        print("no corrupted files found.")
        return

    print(f"found {len(corrupted)} corrupted file(s):")
    for path, error in corrupted:
        size_kb = path.stat().st_size / 1024
        print(f"  {path.name} ({size_kb:.1f} KB): {error}")

    if delete:
        for path, _ in corrupted:
            path.unlink()
        print(f"\ndeleted {len(corrupted)} corrupted file(s). "
              "Re-run the crop pipeline to regenerate them.")
    else:
        print("\npass --delete to remove these, then re-run the crop pipeline "
              "to regenerate just these files.")


if __name__ == "__main__":
    main()
