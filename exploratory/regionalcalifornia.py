import xarray as xr
import geopandas as gpd
import pandas as pd
import cartopy.io.shapereader as shpreader

NC_PATH = "../dataset/california_monthly.nc"
OUT_PATH = "../dataset/california_regional_timeseries.nc"
REGIONS_URL = "https://api.cal-adapt.org/api/climregions/"
SOUTH_COAST_LABEL = "South Coast"


def get_california_regions_gdf():
    print("Fetching and processing climate regions...")
    regions_gdf = gpd.read_file(REGIONS_URL)
    if regions_gdf.crs is None:
        regions_gdf = regions_gdf.set_crs("EPSG:4326")
    regions_gdf = regions_gdf.to_crs("EPSG:4326")

    candidate_cols = ["name", "title", "label", "region", "abbr", "code"]
    region_col = next((c for c in candidate_cols if c in regions_gdf.columns), None)
    if region_col is None:
        non_geom_cols = [c for c in regions_gdf.columns if c != "geometry"]
        region_col = non_geom_cols[0]

    regions_gdf["ClimateRegion"] = regions_gdf[region_col].astype("string").str.strip()
    regions_gdf.loc[regions_gdf["ClimateRegion"].isin(["", "NA", "N/A", "None"]), "ClimateRegion"] = pd.NA

    ne_path = shpreader.natural_earth(resolution="50m", category="cultural", name="admin_1_states_provinces")
    ca_geom = None
    for rec in shpreader.Reader(ne_path).records():
        attrs = rec.attributes
        if attrs.get("admin") == "United States of America" and attrs.get("name") == "California":
            ca_geom = rec.geometry
            break

    regions_filled = regions_gdf.copy()
    regions_filled["ClimateRegion"] = regions_filled["ClimateRegion"].fillna(SOUTH_COAST_LABEL)

    try:
        regions_union = regions_filled.geometry.union_all()
    except AttributeError:
        regions_union = regions_filled.geometry.unary_union

    gap = ca_geom.difference(regions_union)
    regions_dissolved = regions_filled.dissolve(by="ClimateRegion", as_index=False)

    south_idx = regions_dissolved.index[regions_dissolved["ClimateRegion"] == SOUTH_COAST_LABEL]
    if len(south_idx) > 0 and not gap.is_empty:
        regions_dissolved.at[south_idx[0], "geometry"] = regions_dissolved.loc[south_idx[0], "geometry"].union(gap)

    regions_dissolved["geometry"] = regions_dissolved["geometry"].intersection(ca_geom)
    regions_dissolved = regions_dissolved[~regions_dissolved.is_empty].reset_index(drop=True)

    return regions_dissolved


def main():
    print(f"Loading {NC_PATH}...")
    ds = xr.open_dataset(NC_PATH)

    print("Stacking spatial dimensions...")
    ds_flat = ds.stack(location=("lat", "lon"))

    print("Mapping grid points...")

    index = ds_flat.get_index("location")
    lats = index.get_level_values("lat")
    lons = index.get_level_values("lon")

    points_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lons, lats),
        index=index,
        crs="EPSG:4326"
    )

    regions_gdf = get_california_regions_gdf()

    print("Mapping points to regions (sjoin)...")
    joined = gpd.sjoin(points_gdf, regions_gdf[["ClimateRegion", "geometry"]], how="left", predicate="within")

    joined = joined[~joined.index.duplicated(keep='first')]

    region_series = joined["ClimateRegion"]

    print("Grouping and averaging...")
    ds_flat = ds_flat.assign_coords(region=("location", region_series))

    ds_land = ds_flat.dropna("location", subset=["region"])

    print(f"Valid land pixels: {len(ds_land.location)}")

    regional_means = ds_land.groupby("region").mean(dim="location")
    print(f"Saving to {OUT_PATH}...")
    regional_means.to_netcdf(OUT_PATH)


if __name__ == "__main__":
    main()
