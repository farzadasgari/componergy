import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon

from componergy.analysis.regions import build_county_to_region_map, compute_regional_mean


def test_majority_area_overlap_assigns_larger_share():
    county = gpd.GeoDataFrame(
        {"CountyName": ["SPLIT COUNTY"]},
        geometry=[Polygon([(-120, 37), (-119, 37), (-119, 38), (-120, 38)])],
        crs="EPSG:4326",
    )
    region_a = Polygon([(-120, 37), (-119.4, 37), (-119.4, 38), (-120, 38)])  # ~70% of county
    region_b = Polygon([(-119.4, 37), (-119, 37), (-119, 38), (-119.4, 38)])  # ~30% of county
    regions = gpd.GeoDataFrame({"Region": ["A", "B"]}, geometry=[region_a, region_b], crs="EPSG:4326")

    mapping = build_county_to_region_map(county, regions)
    assert mapping.loc[0, "Region"] == "A"


def test_unmapped_county_raises_value_error():
    orphan_county = gpd.GeoDataFrame(
        {"CountyName": ["FAR AWAY COUNTY"]},
        geometry=[Polygon([(-200, 60), (-199, 60), (-199, 61), (-200, 61)])],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"Region": ["A"]},
        geometry=[Polygon([(-120, 37), (-119, 37), (-119, 38), (-120, 38)])],
        crs="EPSG:4326",
    )
    try:
        build_county_to_region_map(orphan_county, regions)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "FAR AWAY COUNTY" in str(e)


def test_regional_mean_correctly_isolates_known_split():
    lats = np.linspace(35, 40, 10)
    lons = np.linspace(-122, -117, 10)
    times = pd.date_range("2010-01-01", periods=3, freq="MS")

    data = np.zeros((3, 10, 10))
    data[:, :, :5] = 10.0
    data[:, :, 5:] = 20.0
    da = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": times, "lat": lats, "lon": lons})

    west_poly = Polygon([(-122, 34), (-119.5, 34), (-119.5, 41), (-122, 41)])
    east_poly = Polygon([(-119.5, 34), (-117, 34), (-117, 41), (-119.5, 41)])
    regions_gdf = gpd.GeoDataFrame({"Region": ["West", "East"]}, geometry=[west_poly, east_poly], crs="EPSG:4326")

    result = compute_regional_mean(da, regions_gdf)
    assert abs(result["West"].mean() - 10.0) < 0.5
    assert abs(result["East"].mean() - 20.0) < 0.5
