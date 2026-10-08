"""Map counties to climate regions, and aggregate gridded climate data to regional means."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask

CA_ALBERS_EQUAL_AREA = "EPSG:3310"


def build_county_to_region_map(counties_gdf: gpd.GeoDataFrame, regions_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Assign each county to the climate region with the largest area overlap."""
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
    """Area-weighted (cosine-latitude) regional mean of a gridded (time, lat, lon) DataArray."""
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


def county_region_shares(counties_gdf: gpd.GeoDataFrame, regions_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Area share of every county in every region it touches (EPSG:3310), one row per (county, region)."""
    counties_proj = counties_gdf[["CountyName", "geometry"]].to_crs(CA_ALBERS_EQUAL_AREA)
    regions_proj = regions_gdf[["Region", "geometry"]].to_crs(CA_ALBERS_EQUAL_AREA)
    pieces = gpd.overlay(counties_proj, regions_proj, how="intersection")
    pieces["area"] = pieces.geometry.area
    pieces = pieces.groupby(["CountyName", "Region"], as_index=False)["area"].sum()
    totals = counties_proj.assign(county_area=counties_proj.geometry.area)[["CountyName", "county_area"]]
    pieces = pieces.merge(totals, on="CountyName")
    pieces["share"] = pieces["area"] / pieces["county_area"]
    return pieces[["CountyName", "Region", "share"]].sort_values(["CountyName", "share"], ascending=[True, False])


def county_split_summary(shares: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Per county: majority region, its area share, and whether the county is materially split."""
    top = shares.sort_values("share", ascending=False).groupby("CountyName").first()
    n = shares.groupby("CountyName")["Region"].nunique()
    out = top.rename(columns={"Region": "Majority_region", "share": "Majority_share"})
    out["N_regions_touched"] = n
    out["Split"] = out["Majority_share"] < threshold
    return out.reset_index().sort_values("Majority_share")
