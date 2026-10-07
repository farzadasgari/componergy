from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

import geopandas as gpd
from shapely.geometry import Polygon

from componergy.download import climate_regions as cr


def _make_fake_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status = lambda: None
    return resp


def _realistic_mercator_wkt(lon_min, lat_min, lon_max, lat_max):
    poly_4326 = Polygon([(lon_min, lat_min), (lon_max, lat_min), (lon_max, lat_max), (lon_min, lat_max)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly_4326], crs="EPSG:4326")
    return gdf.to_crs("EPSG:3857").geometry.iloc[0].wkt


def test_pagination_follows_next_link_until_exhausted():
    page1 = {"features": [{"properties": {"name": "A"}, "geometry": _realistic_mercator_wkt(-121, 38, -120, 39)}],
             "next": "https://api.cal-adapt.org/api/climregions/?page=2&pagesize=2000"}
    page2 = {"features": [{"properties": {"name": "B"}, "geometry": _realistic_mercator_wkt(-119, 36, -118, 37)}],
             "next": None}
    call_count = [0]

    def fake_get(url, timeout=60):
        call_count[0] += 1
        return _make_fake_response(page1 if call_count[0] == 1 else page2)

    cache_path = Path(tempfile.mkdtemp()) / "regions.geojson"
    with patch.object(cr.requests, "get", side_effect=fake_get):
        gdf = cr.get_california_climate_regions(cache_path=cache_path)

    assert call_count[0] == 2
    assert len(gdf) == 2


def test_both_wkt_and_geojson_geometry_formats_parsed():
    wkt_geom = _realistic_mercator_wkt(-121.6, 38.4, -121.3, 38.7)
    poly_4326 = Polygon([(-119, 36), (-118, 36), (-118, 37), (-119, 37)])
    geojson_geom = \
    gpd.GeoDataFrame({"id": [1]}, geometry=[poly_4326], crs="EPSG:4326").to_crs("EPSG:3857").geometry.iloc[
        0].__geo_interface__

    payload = {"features": [
        {"properties": {"name": "WKT Region"}, "geometry": wkt_geom},
        {"properties": {"name": "GeoJSON Region"}, "geometry": geojson_geom},
    ], "next": None}

    cache_path = Path(tempfile.mkdtemp()) / "regions.geojson"
    with patch.object(cr.requests, "get", return_value=_make_fake_response(payload)):
        gdf = cr.get_california_climate_regions(cache_path=cache_path)

    assert set(gdf["Region"]) == {"WKT Region", "GeoJSON Region"}
    bounds = gdf[gdf["Region"] == "WKT Region"].geometry.iloc[0].bounds
    assert abs(bounds[0] - (-121.6)) < 0.01


def test_region_rename_applied():
    payload = {"features": [
        {"properties": {"name": "Sonora Desert"}, "geometry": _realistic_mercator_wkt(-116, 33, -115, 34)}],
               "next": None}
    cache_path = Path(tempfile.mkdtemp()) / "regions.geojson"
    with patch.object(cr.requests, "get", return_value=_make_fake_response(payload)):
        gdf = cr.get_california_climate_regions(cache_path=cache_path)
    assert gdf["Region"].iloc[0] == "Sonoran Desert"


def test_cache_prevents_second_network_call():
    payload = {"features": [
        {"properties": {"name": "Cached Region"}, "geometry": _realistic_mercator_wkt(-121, 38, -120, 39)}],
               "next": None}
    call_count = [0]

    def fake_get(url, timeout=60):
        call_count[0] += 1
        return _make_fake_response(payload)

    cache_path = Path(tempfile.mkdtemp()) / "regions.geojson"
    with patch.object(cr.requests, "get", side_effect=fake_get):
        cr.get_california_climate_regions(cache_path=cache_path)
        cr.get_california_climate_regions(cache_path=cache_path)

    assert call_count[0] == 1
