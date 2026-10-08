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


def allocation_sensitivity(consumption_long_df: pd.DataFrame, shares: pd.DataFrame) -> dict:
    """Regional consumption under majority assignment vs area-weighted allocation."""
    from componergy.figures.composite_stats import deseasonalize_monthly

    county_month = consumption_long_df.groupby(["Date", "CountyName"])["Consumption (MWh)"].sum().reset_index()
    majority = shares.sort_values("share", ascending=False).groupby("CountyName").first()["Region"]
    maj = county_month.assign(Region=county_month["CountyName"].map(majority))
    maj_reg = maj.groupby(["Date", "Region"])["Consumption (MWh)"].sum().unstack()
    w = county_month.merge(shares, on="CountyName")
    w["alloc"] = w["Consumption (MWh)"] * w["share"]
    area_reg = w.groupby(["Date", "Region"])["alloc"].sum().unstack()
    regions = maj_reg.columns.union(area_reg.columns)
    maj_reg = maj_reg.reindex(columns=regions).fillna(0.0)
    area_reg = area_reg.reindex(index=maj_reg.index, columns=regions).fillna(0.0)

    moved = float((maj_reg - area_reg).abs().sum().sum() / 2.0 / maj_reg.sum().sum())
    rows = []
    for region in regions:
        a, b = maj_reg[region], area_reg[region]
        if len(a) < 24:
            continue
        idx = pd.DatetimeIndex(a.index)
        da = deseasonalize_monthly(pd.Series(a.values, index=idx))
        db = deseasonalize_monthly(pd.Series(b.values, index=idx))
        degenerate = a.std() == 0 or b.std() == 0
        rows.append({"Region": region,
                     "corr_deseasonalized": float(da.corr(db)) if not degenerate else np.nan,
                     "level_ratio_area_over_majority": float(b.mean() / a.mean()) if a.mean() else np.nan,
                     "note": ("receives no consumption under one of the two rules" if degenerate else "")})
    return {"share_of_consumption_changing_region": moved, "per_region": pd.DataFrame(rows)}
