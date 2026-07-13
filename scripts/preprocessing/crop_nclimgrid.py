import xarray as xr
import geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "data" / "raw" / "noaa-nclimgrid"
DST_DIR = ROOT / "data" / "processed" / "noaa-nclimgrid-ca"
DST_DIR.mkdir(exist_ok=True, parents=True)

# Source: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
SHAPE_FILE = ROOT / "data" / "raw" / "census-shape" / "tl_2025_us_state.shp"
US = gpd.read_file(SHAPE_FILE)
CA = US[US['NAME'] == 'California'].to_crs("EPSG:4326")

for nc_file in SRC_DIR.glob("*.nc"):
    print(f"Cropping → {nc_file.name}")
    ds = xr.open_dataset(nc_file, chunks={"time": 365})
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    ds_ca = ds.rio.clip(CA.geometry, CA.crs)
    out_name = f"ca-{nc_file.name}"
    out_path = DST_DIR / out_name
    ds_ca.to_netcdf(out_path)
