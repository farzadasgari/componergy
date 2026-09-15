from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask

CA_ALBERS_EQUAL_AREA = "EPSG:3310"


def build_county_to_region_map(counties_gdf: gpd.GeoDataFrame, regions_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    counties_proj = counties_gdf.to_crs(CA_ALBERS_EQUAL_AREA)
    regions_proj = regions_gdf.to_crs(CA_ALBERS_EQUAL_AREA)

    overlaps = gpd.overlay(counties_proj, regions_proj[["Region", "geometry"]], how="intersection")
    overlaps["area"] = overlaps.geometry.area

    best_idx = overlaps.groupby("CountyName")["area"].idxmax()
    mapping = overlaps.loc[best_idx, ["CountyName", "Region"]].reset_index(drop=True)

    unmapped = sorted(set(counties_gdf["CountyName"]) - set(mapping["CountyName"]))
    if unmapped:
        raise ValueError(f"Unmapped counties (check geometry/join): {unmapped}")

    return mapping


def compute_regional_mean(da: xr.DataArray, regions_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    lat = da["lat"]
    lon = da["lon"]

    regions_gdf = regions_gdf.copy()
    regions_gdf["geometry"] = regions_gdf["geometry"].buffer(0)

    regions = regionmask.Regions(
        outlines=list(regions_gdf.geometry),
        names=list(regions_gdf["Region"]),
        abbrevs=[f"R{i + 1:02d}" for i in range(len(regions_gdf))],
        overlap=True,
    )

    mask_3d = regions.mask_3D(lon, lat, drop=False)

    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.mean()

    result = {}
    for name in regions_gdf["Region"]:
        region_mask = mask_3d.sel(names=name)
        masked = da.where(region_mask)
        regional_mean = masked.weighted(weights).mean(dim=["lat", "lon"], skipna=True)
        result[name] = regional_mean.to_series()

    return pd.DataFrame(result).sort_index()
