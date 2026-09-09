"""
Crop raw NOAA nClimGrid-Daily files to the California state boundary.

Reads every raw file in RAW_DATA_CLIMATE_DIR (CONUS-wide) and writes a
California-clipped copy to NOAA_CA_DIR, one file in -> one file out.
Files already cropped are skipped, so re-running this after a fresh
download only processes newly-downloaded months.

Assumes nClimGrid-Daily's native grid is unprojected WGS84 (EPSG:4326)
lat/lon, matching NOAA's published grid specification for this product --
write_crs() below only tags this as metadata, it does not reproject
anything. Spatial dims are explicitly declared as ("lon", "lat") since
rioxarray only auto-detects "x"/"y" by default and otherwise raises
MissingSpatialDimensionError on this data's native dimension names.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import rioxarray
import xarray as xr
from tqdm import tqdm

from componergy.download.boundaries import get_california_boundary
from componergy.paths import NOAA_CA_DIR, RAW_DATA_CLIMATE_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def crop_file(nc_file, california: gpd.GeoDataFrame):
    """
    Crop a single raw NetCDF file to the California boundary.

    Returns the output path.
    """
    ds = xr.open_dataset(nc_file, chunks={"time": 365})
    ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    ds_ca = ds.rio.clip(california.geometry, california.crs)

    out_path = NOAA_CA_DIR / f"ca-{nc_file.name}"
    ds_ca.to_netcdf(out_path)
    return out_path


def main() -> None:
    ensure_dirs()

    print("loading California boundary...")
    california = get_california_boundary()
    print("California boundary ready.")

    raw_files = sorted(RAW_DATA_CLIMATE_DIR.glob("*.nc"))
    pending = [f for f in raw_files if not (NOAA_CA_DIR / f"ca-{f.name}").exists()]
    already_done = len(raw_files) - len(pending)

    print(f"found {len(raw_files)} raw files, {len(pending)} pending, {already_done} already cropped")

    succeeded = 0
    failed: list[str] = []

    for nc_file in tqdm(pending, desc="cropping to California", unit="file"):
        try:
            crop_file(nc_file, california)
            succeeded += 1
        except Exception:
            failed.append(nc_file.name)
            logger.exception("failed to crop %s", nc_file.name)

    print(f"crop complete: {succeeded} succeeded, {len(failed)} failed, {already_done} already done.")
    if failed:
        print("failed files:")
        for name in failed:
            print(f"  - {name}")


if __name__ == "__main__":
    main()