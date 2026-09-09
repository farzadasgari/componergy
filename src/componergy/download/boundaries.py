from __future__ import annotations
import zipfile

import geopandas as gpd

from componergy.paths import CA_BOUNDARY_DIR
from componergy.download.http import download_file
from componergy.download.manifest import append_manifest

CENSUS_CB_STATE_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_500k.zip"
_ZIP_DESTINATION = CA_BOUNDARY_DIR / "cb_2020_us_state_500k.zip"
_EXTRACT_DIR = CA_BOUNDARY_DIR / "cb_2020_us_state_500k"


def get_california_boundary() -> gpd.GeoDataFrame:
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
