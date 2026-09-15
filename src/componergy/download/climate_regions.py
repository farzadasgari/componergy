from __future__ import annotations

import geopandas as gpd
import requests
from shapely import wkt
from shapely.geometry import shape

from componergy.paths import CA_CLIMATE_REGIONS_DIR
