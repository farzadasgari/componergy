from componergy.paths import (
    NOAA_RAW_DIR,
    NOAA_CA_DIR,
    STATE_SHAPEFILE,
)
import sys
from pathlib import Path

import geopandas as gpd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


NOAA_CA_DIR.mkdir(parents=True, exist_ok=True)

# Source: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
US = gpd.read_file(STATE_SHAPEFILE)
CA = US[US['NAME'] == 'California'].to_crs("EPSG:4326")

for nc_file in NOAA_RAW_DIR.glob("*.nc"):
    print(f"Cropping → {nc_file.name}")
    ds = xr.open_dataset(nc_file, chunks={"time": 365})
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    ds_ca = ds.rio.clip(CA.geometry, CA.crs)
    out_name = f"ca-{nc_file.name}"
    out_path = NOAA_CA_DIR / out_name
    ds_ca.to_netcdf(out_path)
