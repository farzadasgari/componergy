"""
Aggregate California-cropped daily NOAA nClimGrid-Daily files into a single
combined monthly-resolution climate dataset spanning the full record.

Monthly aggregation:
    tavg, tmax, tmin -> mean of daily values within the month
    prcp -> sum of daily values within the month

A cell-month is set to NaN, rather than silently aggregated from a partial
month, if fewer than MIN_VALID_DAY_FRACTION of that month's days have
valid data for that variable.
"""

from __future__ import annotations

import logging

import xarray as xr
from tqdm import tqdm

from componergy.paths import NOAA_CA_DIR, NOAA_MONTHLY_FILE, ensure_dirs

logger = logging.getLogger(__name__)

MEAN_VARS = ("tavg", "tmax", "tmin")
SUM_VARS = ("prcp",)
MIN_VALID_DAY_FRACTION = 0.9


def aggregate_file(nc_file) -> xr.Dataset:
    """Aggregate a single California-cropped daily file to one monthly time step."""
    ds = xr.open_dataset(nc_file)
    n_days = ds.sizes["time"]
    min_valid_days = max(1, int(MIN_VALID_DAY_FRACTION * n_days))

    mean_part = ds[list(MEAN_VARS)].resample(time="MS").mean(skipna=True)
    valid_counts = ds[list(MEAN_VARS)].notnull().resample(time="MS").sum()
    for var in MEAN_VARS:
        mean_part[var] = mean_part[var].where(valid_counts[var] >= min_valid_days)

    sum_part = ds[list(SUM_VARS)].resample(time="MS").sum(min_count=min_valid_days)

    return xr.merge([mean_part, sum_part])


def main(force: bool = False) -> None:
    ensure_dirs()

    if NOAA_MONTHLY_FILE.exists() and not force:
        print(f"{NOAA_MONTHLY_FILE} already exists, skipping (pass force=True to rebuild).")
        return

    raw_files = sorted(NOAA_CA_DIR.glob("*.nc"))
    print(f"found {len(raw_files)} cropped daily files to aggregate")

    monthly_slices = []
    failed: list[str] = []
    for nc_file in tqdm(raw_files, desc="aggregating to monthly", unit="file"):
        try:
            monthly_slices.append(aggregate_file(nc_file))
        except Exception:
            failed.append(nc_file.name)
            logger.exception("failed to aggregate %s", nc_file.name)

    if not monthly_slices:
        print("no files aggregated, nothing to write.")
        return

    combined = xr.concat(monthly_slices, dim="time").sortby("time")
    combined.to_netcdf(
        NOAA_MONTHLY_FILE,
        encoding={v: {"zlib": True, "complevel": 5} for v in combined.data_vars},
    )

    print(
        f"wrote {NOAA_MONTHLY_FILE}: {combined.sizes['time']} months, "
        f"{combined.sizes['lat']}x{combined.sizes['lon']} grid"
    )
    if failed:
        print(f"{len(failed)} files failed to aggregate:")
        for name in failed:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
