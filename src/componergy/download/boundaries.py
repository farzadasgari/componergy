"""
Authoritative geographic boundaries for spatial analysis.

California's boundary is sourced from the U.S. Census Bureau's
cartographic boundary files.
"""

from __future__ import annotations
import zipfile

import geopandas as gpd

from componergy.paths import CA_BOUNDARY_DIR, CA_COUNTIES_DIR
from componergy.download.http import download_file
from componergy.download.manifest import append_manifest

CENSUS_CB_STATE_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_500k.zip"
_ZIP_DESTINATION = CA_BOUNDARY_DIR / "cb_2020_us_state_500k.zip"
_EXTRACT_DIR = CA_BOUNDARY_DIR / "cb_2020_us_state_500k"


def get_california_boundary() -> gpd.GeoDataFrame:
    """
    Return California's boundary as a one-row GeoDataFrame in EPSG:4326.

    Downloads and caches the Census cartographic boundary state file
    on first use; later calls just read the cached shapefile.
    """
    if not _EXTRACT_DIR.exists():
        _, digest = download_file(
            dataset_name="Census Cartographic Boundary - States (1:500k)",
            source_page="https://www.census.gov/geographies/mapping-files/2020/geo/carto-boundary-file.html",
            url=CENSUS_CB_STATE_URL,
            destination=_ZIP_DESTINATION,
        )
        append_manifest(
            name="Census Cartographic Boundary - States (1:500k)",
            url=CENSUS_CB_STATE_URL,
            destination=_ZIP_DESTINATION,
            digest=digest,
        )
        _EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(_ZIP_DESTINATION) as zf:
            zf.extractall(_EXTRACT_DIR)

    shp_path = next(_EXTRACT_DIR.glob("*.shp"))
    us_states = gpd.read_file(shp_path)
    california = us_states[us_states["NAME"]
                           == "California"].to_crs("EPSG:4326")
    return california.reset_index(drop=True)


# appended to boundaries.py, after get_california_boundary()

CENSUS_CB_COUNTY_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip"
_COUNTY_ZIP_DESTINATION = CA_COUNTIES_DIR / "cb_2020_us_county_500k.zip"
_COUNTY_EXTRACT_DIR = CA_COUNTIES_DIR / "cb_2020_us_county_500k"


def get_california_counties() -> gpd.GeoDataFrame:
    """
    Return California's counties as a GeoDataFrame (CountyName, geometry) in EPSG:4326.

    Same 2020 cartographic-boundary vintage as get_california_boundary(),
    for consistency between the state and county boundaries used across
    the pipeline. Downloads and caches on first use.
    """
    if not _COUNTY_EXTRACT_DIR.exists():
        _, digest = download_file(
            dataset_name="Census Cartographic Boundary - Counties (1:500k)",
            source_page="https://www.census.gov/geographies/mapping-files/2020/geo/carto-boundary-file.html",
            url=CENSUS_CB_COUNTY_URL,
            destination=_COUNTY_ZIP_DESTINATION,
        )
        append_manifest(
            name="Census Cartographic Boundary - Counties (1:500k)",
            url=CENSUS_CB_COUNTY_URL,
            destination=_COUNTY_ZIP_DESTINATION,
            digest=digest,
        )
        _COUNTY_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(_COUNTY_ZIP_DESTINATION) as zf:
            zf.extractall(_COUNTY_EXTRACT_DIR)

    shp_path = next(_COUNTY_EXTRACT_DIR.glob("*.shp"))
    all_counties = gpd.read_file(shp_path)
    california_counties = all_counties[all_counties["STATEFP"] == "06"].to_crs("EPSG:4326")
    california_counties = california_counties.copy()
    california_counties["CountyName"] = california_counties["NAME"].astype("string").str.upper().str.strip()
    return california_counties[["CountyName", "geometry"]].reset_index(drop=True)
