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


def _fetch_all_pages(url, pagesize=2000):
    sep = "&" if "?" in url else "?"
    next_url = f"{url}{sep}pagesize={pagesize}"
    features = []
    while next_url:
        response = requests.get(next_url, timeout=60)
        response.raise_for_status()
        data = response.json()
        features.extend(data.get("features", []))
        next_url = data.get("next")
    return features


def _feature_to_geometry(feature):
    geom = feature.get("geometry")
    if geom is None:
        return None
    if isinstance(geom, str):
        return wkt.loads(geom)
    if isinstance(geom, dict):
        return shape(geom)
    return None


def _features_to_gdf(features):
    rows = []
    for f in features:
        geom = _feature_to_geometry(f)
        if geom is None:
            continue
        rows.append({**(f.get("properties") or {}), "geometry": geom})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry")
    return gdf.set_crs("EPSG:3857", allow_override=True).to_crs("EPSG:4326")
