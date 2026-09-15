from __future__ import annotations

import geopandas as gpd
import requests
from shapely import wkt
from shapely.geometry import shape

from componergy.paths import CA_CLIMATE_REGIONS_DIR

CLIMATE_REGIONS_URL = "https://api.cal-adapt.org/api/climregions/"
REGION_RENAME = {"Sonora Desert": "Sonoran Desert"}
_NAME_COLUMN_CANDIDATES = ["name", "title", "label", "region", "abbr", "code"]
_CACHE_PATH = CA_CLIMATE_REGIONS_DIR / "climate_regions.geojson"
