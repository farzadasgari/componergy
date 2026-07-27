import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import geopandas as gpd
import requests

CSV_PATH = Path("../dataset/impact_analysis_results_per_region.csv")
OUT_DIR = Path("../outputs")
MAP_DIR = OUT_DIR / "map"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
value_col = "Mean_AnomPct_vsMeanBase"
df[value_col] = pd.to_numeric(df[value_col], errors="coerce").clip(-100, 100)

event_map = {
    "Heatwave_only": "Heatwave",
    "Drought_only": "Drought",
    "Compound": "Compound",
    "Heatwave": "Heatwave",
    "Drought": "Drought",
}

colors = {"Heatwave": "#d73027", "Drought": "#4575b4", "Compound": "#7b3294"}
order = ["Heatwave", "Drought", "Compound"]

impact_df = df[df["Event"] != "Normal"].copy()
impact_df["Event_Group"] = impact_df["Event"].map(
    event_map).fillna(impact_df["Event"])

g = (
    impact_df.groupby(["Region", "Class", "Case", "Event_Group"])[value_col]
    .mean()
    .reset_index()
)

percs = (
    g.pivot_table(
        index=["Region", "Class", "Case"],
        columns="Event_Group",
        values=value_col,
        aggfunc="mean",
    )
    .reset_index()
)

for c in order:
    if c not in percs.columns:
        percs[c] = np.nan

url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
geojson = requests.get(url).json()
ca_features = [f for f in geojson["features"]
               if str(f.get("id", "")).startswith("06")]
for f in ca_features:
    f["properties"]["id"] = str(f["id"]).zfill(5)

ca_gdf = gpd.GeoDataFrame.from_features(ca_features, crs="EPSG:4326")
ca_gdf["id"] = ca_gdf["id"].astype(str).str.zfill(5)

region_counties = {
    "Central Coast": ["06069", "06079", "06083", "06087", "06111", "06053"],
    "Mojave Desert": ["06027", "06029", "06071"],
    "North Central": ["06001", "06013", "06075", "06081", "06085", "06095", "06097", "06041", "06055"],
    "North Coast": ["06015", "06023", "06033", "06045", "06105"],
    "Northeast": ["06035", "06049", "06093", "06089", "06063"],
    "Sacramento-Delta": ["06007", "06011", "06017", "06021", "06067", "06077", "06101", "06113", "06115"],
    "San Joaquin Valley": ["06019", "06031", "06039", "06047", "06099", "06107", "06077"],
    "Sierra": ["06003", "06005", "06009", "06043", "06057", "06061", "06063", "06091", "06109"],
    "Sonora Desert": ["06025", "06065", "06071"],
    "South Coast": ["06037", "06059", "06065", "06071", "06073"],
}

fips_to_region = {fips: region for region,
                  fips_list in region_counties.items() for fips in fips_list}
ca_gdf["Region"] = ca_gdf["id"].map(fips_to_region).fillna("Unassigned")

regions_gdf = ca_gdf.dissolve(by="Region", as_index=False)
regions_gdf = regions_gdf.to_crs("EPSG:3310")
regions_gdf["pt"] = regions_gdf.geometry.representative_point()
regions_gdf["no_glyph"] = regions_gdf["Region"].str.upper(
).str.contains("MOJAVE|SONORA|UNASSIGNED", regex=True)
regions_gdf["is_desert"] = regions_gdf["Region"].str.upper(
).str.contains("MOJAVE|SONORA", regex=True)

sectors = ["Residential", "Commercial", "Industrial", "Agricultural"]
cases = ["Pure", "Overlapping"]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
plt.subplots_adjust(wspace=0.02, hspace=0.05)

w, h = 0.3, 0.1
pad = 0.006

for i, case in enumerate(cases):
    for j, sector in enumerate(sectors):
        ax = axes[i, j]

        regions_gdf[regions_gdf["no_glyph"] & ~regions_gdf["is_desert"]].plot(
            ax=ax, color="#AFAFAF", edgecolor="white", linewidth=3, alpha=1.0
        )
        regions_gdf[regions_gdf["is_desert"]].plot(
            ax=ax, color="#AFAFAF", edgecolor="white", linewidth=3, alpha=0.95
        )
        regions_gdf[~regions_gdf["no_glyph"]].plot(
            ax=ax, color="#D3D3D3", edgecolor="white", linewidth=3, alpha=1.0
        )

        ax.set_axis_off()
        # ax.set_title(f"{case} • {sector}", fontsize=11)
        ax.set_title(f"{sector}", fontsize=14)

        sub = percs[(percs["Case"] == case) & (
            percs["Class"] == sector)].copy()

        for _, r in regions_gdf.iterrows():
            if bool(r["no_glyph"]):
                continue

            region = r["Region"]
            pt = r["pt"]
            row = sub[sub["Region"] == region]
            if row.empty:
                continue

            vals = row.iloc[0][order].to_numpy(dtype=float)
            if np.all(np.isnan(vals)):
                continue

            x, y = float(pt.x), float(pt.y)
            ax_xy = ax.transAxes.inverted().transform(ax.transData.transform((x, y)))
            ax_x, ax_y = float(ax_xy[0]), float(ax_xy[1])

            x0, y0 = ax_x - w / 5, ax_y - h / 5
            x0 = min(max(x0, pad), 1 - w - pad)
            y0 = min(max(y0, pad), 1 - h - pad)

            iax = ax.inset_axes([x0, y0, w, h], transform=ax.transAxes)
            iax.set_facecolor((1, 1, 1, 0.88))

            ypos = np.array([2, 1, 0], dtype=float)
            iax.axvline(0, linewidth=0.7, alpha=0.25)
            iax.barh(
                ypos,
                np.nan_to_num(vals, nan=0.0),
                height=0.95,
                color=[colors[e] for e in order],
                edgecolor="black",
                linewidth=1,
            )

            iax.set_xlim(-10, 110)
            iax.set_ylim(-0.6, 2.6)
            iax.axis("off")

legend_elements = [
    Patch(facecolor=colors[e], edgecolor="none", label=e) for e in order]
fig.legend(handles=legend_elements, loc="lower center", ncols=3, frameon=False, fontsize=18)

fig.savefig(MAP_DIR / "california_extreme_event_impact_glyphmaps.png",
            dpi=400, bbox_inches="tight")
fig.savefig(MAP_DIR / "california_extreme_event_impact_glyphmaps.pdf",
            bbox_inches="tight")
plt.close(fig)
