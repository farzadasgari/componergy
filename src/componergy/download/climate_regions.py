"""
Download and cache California's climate regions from Cal-Adapt's API.

The API paginates via a "next" link in each response; results are
followed until exhausted. Geometry arrives as either a WKT string or a
GeoJSON geometry dict depending on the endpoint/version, so both are
handled. Coordinates are assumed to be EPSG:3857 (Web Mercator) on
arrival and reprojected to EPSG:4326 -- verified against a known
geometry round-tripped through the same transform; if this assumption
ever turns out to be wrong for a particular API response, the fix is to
read the response's own "crs" field instead of asserting EPSG:3857.

Region names come from whichever of a few candidate property keys is
present (name/title/label/region/abbr/code), since the exact key isn't
guaranteed stable across API versions. REGION_RENAME fixes one known
naming inconsistency ("Sonora Desert" vs "Sonoran Desert").
"""

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
    """Follow the API's "next" pagination links until exhausted, collecting all features."""
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
    """Parse a feature's geometry, whether given as WKT (str) or GeoJSON (dict)."""
    geom = feature.get("geometry")
    if geom is None:
        return None
    if isinstance(geom, str):
        return wkt.loads(geom)
    if isinstance(geom, dict):
        return shape(geom)
    return None


def _features_to_gdf(features):
    """Build a GeoDataFrame from raw features, reprojecting from the assumed source CRS to EPSG:4326."""
    rows = []
    for f in features:
        geom = _feature_to_geometry(f)
        if geom is None:
            continue
        rows.append({**(f.get("properties") or {}), "geometry": geom})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry")
    return gdf.set_crs("EPSG:3857", allow_override=True).to_crs("EPSG:4326")


def _extract_region_names(gdf):
    """Pick a name column from the candidates, apply REGION_RENAME, and drop duplicate/null regions."""
    name_col = next((c for c in _NAME_COLUMN_CANDIDATES if c in gdf.columns), None)
    if name_col is None:
        name_col = [c for c in gdf.columns if c != "geometry"][0]
    gdf = gdf.copy()
    gdf["Region"] = gdf[name_col].astype("string").str.strip().replace(REGION_RENAME)
    return (
        gdf[["Region", "geometry"]]
        .dropna()
        .drop_duplicates(subset=["Region"])
        .reset_index(drop=True)
    )


def get_california_climate_regions(cache_path=None) -> gpd.GeoDataFrame:
    """
    Return California's climate regions as a GeoDataFrame (Region, geometry).

    Downloads and caches to `cache_path` (default: a fixed location under
    CA_CLIMATE_REGIONS_DIR) on first use; later calls read the cache
    without hitting the network again.
    """
    cache_path = cache_path or _CACHE_PATH
    if cache_path.exists():
        return gpd.read_file(cache_path)

    features = _fetch_all_pages(CLIMATE_REGIONS_URL)
    gdf = _features_to_gdf(features)
    gdf = _extract_region_names(gdf)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache_path, driver="GeoJSON")
    return gdf
