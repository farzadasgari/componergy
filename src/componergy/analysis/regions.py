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
