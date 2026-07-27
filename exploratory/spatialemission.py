import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
import matplotlib
import cartopy.io.shapereader as shpreader

CSV_PATH = "../dataset/emission/csvs/SO2_2007.csv"
STATE_CODE_CA = "06"
PLOT_EXTENT = [-125, -113, 32, 42.5]
REGIONS_URL = "https://api.cal-adapt.org/api/climregions/"
OUT_FIG_PNG = "../plots/results/california_emission_so2.png"
SOUTH_COAST_LABEL = "South Coast"

df = pd.read_csv(
    CSV_PATH,
    dtype={
        "State Code": "string",
        "County Code": "string",
        "Site Num": "string",
        "Parameter Code": "string",
        "POC": "Int64",
    },
    parse_dates=["Date Local", "Date of Last Change"],
)

ca = df[df["State Code"] == STATE_CODE_CA].copy()

station_id_cols = ["State Code", "County Code", "Site Num"]
keep_cols = station_id_cols + [
    "Latitude", "Longitude",
    "Local Site Name", "Address",
    "City Name", "County Name", "CBSA Name"
]

stations_ca = (
    ca[keep_cols]
    .dropna(subset=["Latitude", "Longitude"])
    .drop_duplicates(subset=station_id_cols)
    .reset_index(drop=True)
)

stations_ca["AQS_Site_ID"] = (
    stations_ca["State Code"].str.zfill(2) + "-" +
    stations_ca["County Code"].str.zfill(3) + "-" +
    stations_ca["Site Num"].str.zfill(4)
)

print(f"Unique CA stations: {len(stations_ca)}")

stations_gdf = gpd.GeoDataFrame(
    stations_ca,
    geometry=gpd.points_from_xy(stations_ca["Longitude"], stations_ca["Latitude"]),
    crs="EPSG:4326",
)

regions_gdf = gpd.read_file(REGIONS_URL)
if regions_gdf.crs is None:
    regions_gdf = regions_gdf.set_crs("EPSG:4326")
regions_gdf = regions_gdf.to_crs("EPSG:4326")

candidate_cols = ["name", "title", "label", "region", "abbr", "code"]
region_col = next((c for c in candidate_cols if c in regions_gdf.columns), None)
if region_col is None:
    non_geom_cols = [c for c in regions_gdf.columns if c != "geometry"]
    if len(non_geom_cols) == 0:
        raise ValueError("No non-geometry columns found in regions layer.")
    region_col = non_geom_cols[0]

regions_gdf["ClimateRegion"] = regions_gdf[region_col].astype("string").str.strip()
regions_gdf.loc[regions_gdf["ClimateRegion"].isin(["", "NA", "N/A", "None"]), "ClimateRegion"] = pd.NA

stations_with_region = gpd.sjoin(
    stations_gdf,
    regions_gdf[["ClimateRegion", "geometry"]],
    how="left",
    predicate="within"
).drop(columns=["index_right"], errors="ignore")

stations_with_region["ClimateRegion"] = stations_with_region["ClimateRegion"].fillna(SOUTH_COAST_LABEL)

print("\nStations per climate region:")
print(stations_with_region["ClimateRegion"].value_counts(dropna=False))

ne_path = shpreader.natural_earth(
    resolution="50m",
    category="cultural",
    name="admin_1_states_provinces"
)
ca_geom = None
for rec in shpreader.Reader(ne_path).records():
    attrs = rec.attributes
    if attrs.get("admin") == "United States of America" and attrs.get("name") == "California":
        ca_geom = rec.geometry
        break
if ca_geom is None:
    raise RuntimeError("Could not find California geometry in Natural Earth admin_1_states_provinces.")

regions_filled = regions_gdf.copy()
regions_filled["ClimateRegion"] = regions_filled["ClimateRegion"].fillna(SOUTH_COAST_LABEL)

regions_union = regions_filled.geometry.union_all()
gap = ca_geom.difference(regions_union)

regions_dissolved = regions_filled.dissolve(by="ClimateRegion", as_index=False)

south_idx = regions_dissolved.index[regions_dissolved["ClimateRegion"] == SOUTH_COAST_LABEL]
if len(south_idx) == 0:
    raise RuntimeError("South Coast region not found in regions layer.")
south_idx = south_idx[0]

if not gap.is_empty:
    regions_dissolved.at[south_idx, "geometry"] = regions_dissolved.loc[south_idx, "geometry"].union(gap)

regions_dissolved["geometry"] = regions_dissolved["geometry"].intersection(ca_geom)
regions_dissolved = regions_dissolved[~regions_dissolved.is_empty].reset_index(drop=True)

region_names = sorted(regions_dissolved["ClimateRegion"].dropna().unique())
cmap = matplotlib.colormaps["tab20"].resampled(len(region_names))
region_to_color = {name: cmap(i) for i, name in enumerate(region_names)}

stations_with_region["plot_color"] = stations_with_region["ClimateRegion"].map(region_to_color)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8)
ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.8)

for _, row in regions_dissolved.iterrows():
    name = row["ClimateRegion"]
    ax.add_geometries(
        [row.geometry],
        crs=ccrs.PlateCarree(),
        facecolor=region_to_color[name],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.18,
        zorder=1
    )

ax.scatter(
    stations_with_region["Longitude"],
    stations_with_region["Latitude"],
    s=32,
    c=stations_with_region["plot_color"].tolist(),
    edgecolors="k",
    linewidths=0.4,
    transform=ccrs.PlateCarree(),
    zorder=2
)

legend_elements = [
    Line2D([0], [0], marker="o", color="w", label=name,
           markerfacecolor=region_to_color[name], markeredgecolor="k", markersize=7)
    for name in region_names
]

ax.legend(
    handles=legend_elements,
    title="Climate Region",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True
)

plt.tight_layout()
plt.savefig(OUT_FIG_PNG, dpi=400, bbox_inches="tight")
plt.show()
