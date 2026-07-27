import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from shapely.geometry import shape
from shapely import wkt
from urllib.request import urlopen
import json
from pathlib import Path
import zipfile
import io
import requests
import matplotlib.patheffects as pe

from scipy.stats import ttest_ind
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from componergy.paths import COUNTY_CONSUMPTION_XLSX, CALIFORNIA_MONTHLY_FILE, FIGURES_DIR

consumption_path = str(COUNTY_CONSUMPTION_XLSX)
netcdf_path = str(CALIFORNIA_MONTHLY_FILE)
regions_url = "https://api.cal-adapt.org/api/climregions/"

OUT_DIR = str(FIGURES_DIR / "Figure 4")
os.makedirs(OUT_DIR, exist_ok=True)

REGION_RENAME = {
"Sonora Desert": "Sonoran Desert",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "axes.linewidth": 1.4,
    "savefig.dpi": 600,
})


def zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean(axis=0)) / df.std(axis=0, ddof=0)


def deseasonalize_monthly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    months = out.index.month
    for c in out.columns:
        clim = out[c].groupby(months).transform("mean")
        out[c] = out[c] - clim
    return out


def to_time_index_from_year_month(da):
    da_1d = da.stack(time=("year", "month")).sortby("time")
    y = da_1d["year"].values
    m = da_1d["month"].values
    idx = pd.to_datetime([f"{int(yy):04d}-{int(mm):02d}-01" for yy, mm in zip(y, m)])
    return da_1d, idx


def fetch_all_caladapt(url, pagesize=2000):
    sep = "&" if "?" in url else "?"
    next_url = f"{url}{sep}pagesize={pagesize}"
    feats, meta = [], None
    while next_url:
        with urlopen(next_url) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        feats.extend(data.get("features", []))
        meta = data
        next_url = data.get("next", None)
    return {"type": "FeatureCollection", "features": feats, "crs": (meta.get("crs") if meta else None)}


def caladapt_to_gdf(fc):
    rows = []
    for f in fc.get("features", []):
        props = f.get("properties", {}) or {}
        geom = f.get("geometry", None)
        if geom is None:
            continue
        if isinstance(geom, str):
            geom_obj = wkt.loads(geom)
        elif isinstance(geom, dict):
            geom_obj = shape(geom)
        else:
            continue
        rows.append({**props, "geometry": geom_obj})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry")
    return gdf.set_crs("EPSG:3857", allow_override=True).to_crs("EPSG:4326")


def load_climate_regions(regions_url):
    fc = fetch_all_caladapt(regions_url, pagesize=2000)
    gdf = caladapt_to_gdf(fc)
    candidate_cols = ["name", "title", "label", "region", "abbr", "code"]
    name_col = next((c for c in candidate_cols if c in gdf.columns), None)
    if name_col is None:
        name_col = [c for c in gdf.columns if c != "geometry"][0]
    gdf["Region"] = (
        gdf[name_col]
        .astype("string")
        .str.strip()
        .replace(REGION_RENAME)
    )
    gdf = (
        gdf[["Region", "geometry"]]
        .dropna()
        .drop_duplicates(subset=["Region"])
        .reset_index(drop=True)
    )
    return gdf


def download_ca_counties_cartographic():
    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
    content = requests.get(url, timeout=120).content
    zf = zipfile.ZipFile(io.BytesIO(content))
    tmp = Path("./Figure 4/_tmp_counties")
    tmp.mkdir(exist_ok=True)
    zf.extractall(tmp)
    shp = next(tmp.glob("*.shp"))
    gdf = gpd.read_file(shp).to_crs("EPSG:4326")
    gdf = gdf[gdf["STATEFP"] == "06"].copy()
    gdf["CountyName"] = gdf["NAME"].astype("string").str.upper().str.strip()
    return gdf[["CountyName", "geometry"]].copy()


def build_county_to_region_map(counties_gdf, regions_gdf, method="area"):
    counties = counties_gdf.copy()
    regions = regions_gdf.copy()

    if method == "point":
        pts = counties.copy()
        pts["geometry"] = pts.geometry.representative_point()
        joined = gpd.sjoin(pts, regions[["Region", "geometry"]], how="left", predicate="within")
        mapping = joined[["CountyName", "Region"]].drop_duplicates()
        if mapping["Region"].isna().any():
            unmapped = mapping[mapping["Region"].isna()]["CountyName"].unique().tolist()
            raise ValueError(f"Unmapped counties (point join): {unmapped[:10]}{'...' if len(unmapped) > 10 else ''}")
        return mapping

    counties_ea = counties.to_crs("EPSG:3310")
    regions_ea = regions.to_crs("EPSG:3310")

    try:
        inter = gpd.overlay(
            counties_ea[["CountyName", "geometry"]],
            regions_ea[["Region", "geometry"]],
            how="intersection"
        )
    except Exception as e:
        print(f"[WARN] Area-overlap mapping failed ({e}). Falling back to point method.")
        return build_county_to_region_map(counties_gdf, regions_gdf, method="point")

    inter["A"] = inter.geometry.area
    best = (inter.sort_values("A", ascending=False)
                 .groupby("CountyName")
                 .first()
                 .reset_index()[["CountyName", "Region"]])

    if best["Region"].isna().any():
        unmapped = best[best["Region"].isna()]["CountyName"].unique().tolist()
        raise ValueError(f"Unmapped counties (area join): {unmapped[:10]}{'...' if len(unmapped) > 10 else ''}")

    return best


def statewide_driver(ds_window: xr.Dataset, varname: str) -> pd.Series:
    """Area-weighted (cos(lat)) mean over CA grid."""
    lat = ds_window["lat"]
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    da = ds_window[varname].weighted(w).mean(dim=["lat", "lon"])
    da_1d, idx = to_time_index_from_year_month(da)
    return pd.Series(da_1d.values, index=idx).sort_index()


def region_abbrev(name: str) -> str:
    """Short labels for map (keeps it clean)."""
    manual = {
        "San Joaquin Valley": "SJV",
        "Sacramento-Delta": "SD",
        "Central Coast": "CC",
        "North Coast": "NC",
        "South Coast": "SC",
        "North Central": "N-C",
        "Southern Interior": "SI",
        "Mojave Desert": "MD",
        "Sonoran Desert": "SoD",
        "Northeast": "NE",
        "Sierra": "SR",
    }
    if name in manual:
        return manual[name]
    parts = [p for p in name.replace("-", " ").split() if p]
    return "".join(p[0].upper() for p in parts[:3])


def text_color_for_bg(value, norm, cmap_obj, light_thresh=0.55):
    """
    Choose black/white based on background (colormap) luminance.
    light_thresh in [0,1]: higher => more likely black text.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "black"
    r, g, b, _ = cmap_obj(norm(value))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if lum >= light_thresh else "white"


df = pd.read_excel(consumption_path, sheet_name="Electricity Consumption")
df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
df["CountyName"] = df["CountyName"].astype("string").str.upper().str.strip()

total_candidates = ["Consumption (MWh)", "Consumption_MWh", "Total Consumption (MWh)", "TOTAL (MWh)"]
total_col = next((c for c in total_candidates if c in df.columns), None)

if total_col is None:
    sector_like = [c for c in df.columns if ("MWh" in str(c)) and (c not in ["Year", "Month", "Date"])]
    sector_like = [c for c in sector_like if "TOTAL" not in str(c).upper() and "CONSUMPTION" not in str(c).upper()]
    if len(sector_like) == 0:
        raise ValueError("Could not find total consumption column or sector MWh columns to sum.")
    df["__TOTAL_MWh__"] = df[sector_like].sum(axis=1, numeric_only=True)
    total_col = "__TOTAL_MWh__"

regions_gdf = load_climate_regions(regions_url)
counties_gdf = download_ca_counties_cartographic()
county_region = build_county_to_region_map(counties_gdf, regions_gdf, method="area")

df = df.merge(county_region, on="CountyName", how="left")
if df["Region"].isna().any():
    missing = df.loc[df["Region"].isna(), "CountyName"].dropna().unique().tolist()
    raise ValueError(f"Counties missing region assignment: {missing[:10]}{'...' if len(missing) > 10 else ''}")

electricity_regional = (
    df.groupby(["Date", "Region"])[total_col].sum()
      .unstack("Region")
      .sort_index()
    / 1000.0
)

ds = xr.open_dataset(netcdf_path)
ds_window = ds.sel(year=slice(2008, 2024))

sti_state = statewide_driver(ds_window, "sti_scdhi")
sapei_state = statewide_driver(ds_window, "sapei_scdhi")
scdhi_state = statewide_driver(ds_window, "scdhi")

start = max(electricity_regional.index.min(), sti_state.index.min(),
            sapei_state.index.min(), scdhi_state.index.min())
end = min(electricity_regional.index.max(), sti_state.index.max(),
          sapei_state.index.max(), scdhi_state.index.max())

electricity_regional = electricity_regional.loc[start:end]
sti_state = sti_state.loc[start:end]
sapei_state = sapei_state.loc[start:end]
scdhi_state = scdhi_state.loc[start:end]

smooth_window = 3
E = electricity_regional.rolling(smooth_window, center=True, min_periods=1).mean()
STI = sti_state.rolling(smooth_window, center=True, min_periods=1).mean()
SAPEI = sapei_state.rolling(smooth_window, center=True, min_periods=1).mean()
SCDHI = scdhi_state.rolling(smooth_window, center=True, min_periods=1).mean()

E_local = deseasonalize_monthly(E)
E_local_z = zscore_frame(E_local)

q_heat = 0.90
q_drought = 0.10
q_compound = 0.10
q_normal = 0.90

thr_heat = STI.quantile(q_heat)
thr_drought = SAPEI.quantile(q_drought)
thr_comp = SCDHI.quantile(q_compound)
thr_norm = SCDHI.quantile(q_normal)

heat = STI >= thr_heat
drought = SAPEI <= thr_drought
compound = SCDHI <= thr_comp

heat_only = heat & (~drought)
drought_only = drought & (~heat)
normal = SCDHI >= thr_norm

event_defs = {
    "Heatwave": heat_only,
    "Drought": drought_only,
    "Compound": compound,
}

N_normal = int(normal.sum())
N_events = {k: int(v.sum()) for k, v in event_defs.items()}

regions_order = [r for r in list(regions_gdf["Region"]) if r in E_local_z.columns]

rows = []
for region in regions_order:
    y_z = E_local_z[region].dropna()
    y_raw = E[region].reindex(y_z.index).dropna()

    norm_mask = normal.reindex(y_z.index).fillna(False)

    y_norm_z = y_z[norm_mask.values]
    y_norm_raw = y_raw[norm_mask.values]

    for ev_name, ev_mask in event_defs.items():
        m = ev_mask.reindex(y_z.index).fillna(False)
        y_ev_z = y_z[m.values]
        y_ev_raw = y_raw[m.values]

        tstat, pval = ttest_ind(y_ev_z.values, y_norm_z.values, equal_var=False, nan_policy="omit")

        mean_ev_z = float(np.nanmean(y_ev_z.values))
        mean_no_z = float(np.nanmean(y_norm_z.values))
        delta_z = mean_ev_z - mean_no_z

        mean_ev_raw = float(np.nanmean(y_ev_raw.values))
        mean_no_raw = float(np.nanmean(y_norm_raw.values))
        delta_pct = 100.0 * (mean_ev_raw - mean_no_raw) / mean_no_raw if mean_no_raw != 0 else np.nan

        rows.append({
            "Region": region,
            "Event": ev_name,
            "N_event": int(m.sum()),
            "N_normal": int(norm_mask.sum()),
            "Mean_event_z": mean_ev_z,
            "Mean_normal_z": mean_no_z,
            "Delta_z": delta_z,
            "Mean_event_GWh": mean_ev_raw,
            "Mean_normal_GWh": mean_no_raw,
            "Delta_pct": delta_pct,
            "tstat": float(tstat),
            "p_value": float(pval),
        })

comp = pd.DataFrame(rows)

comp.to_csv(os.path.join(OUT_DIR, "figure 4 - event composites.csv"), index=False)
with open(os.path.join(OUT_DIR, "figure 4 - event composites.log"), "w", encoding="utf-8") as f:
    f.write("=== Figure 4: Zonal consumption event composites ===\n\n")
    f.write(f"Period: {start.date()} to {end.date()}  | months={len(E.index)}\n")
    f.write(f"Smoothing: {smooth_window}-month centered rolling mean\n")
    f.write("Consumption anomalies: deseasonalized monthly climatology removed, then z-scored per region\n\n")
    f.write("Event thresholds (statewide):\n")
    f.write(f"  Heat: STI ≥ q{q_heat:.2f} (thr={thr_heat:.3f})\n")
    f.write(f"  Drought: SAPEI ≤ q{q_drought:.2f} (thr={thr_drought:.3f})\n")
    f.write(f"  Compound-index: SCDHI ≤ q{q_compound:.2f} (thr={thr_comp:.3f})\n")
    f.write(f"  Normal: SCDHI ≥ q{q_normal:.2f} (thr={thr_norm:.3f})\n\n")
    f.write(f"Counts: Normal={N_normal} | " + " | ".join([f"{k}={v}" for k, v in N_events.items()]) + "\n\n")

state_outline = counties_gdf.dissolve().boundary

regions_plot = regions_gdf.copy()
regions_plot["abbr"] = regions_plot["Region"].map(region_abbrev)
regions_plot["centroid"] = regions_plot.geometry.representative_point()


def attach_event_to_gdf(event_name):
    sub = comp[comp["Event"] == event_name].set_index("Region")
    g = regions_plot.copy()
    g["Delta_pct"] = g["Region"].map(sub["Delta_pct"].to_dict())
    g["p_value"] = g["Region"].map(sub["p_value"].to_dict())
    return g


delta_pct_all = comp["Delta_pct"].replace([np.inf, -np.inf], np.nan).dropna().values
vmax_pct = np.nanpercentile(np.abs(delta_pct_all), 90) if len(delta_pct_all) else 10.0
vmax_pct = max(5.0, float(vmax_pct))

delta_z_all = comp["Delta_z"].replace([np.inf, -np.inf], np.nan).dropna().values
vmax_z = np.nanpercentile(np.abs(delta_z_all), 90) if len(delta_z_all) else 1.0
vmax_z = max(0.5, float(vmax_z))

events_for_fig = ["Heatwave", "Drought", "Compound"]

fig = plt.figure(figsize=(14.8, 10.2), constrained_layout=False)
gs = GridSpec(nrows=2, ncols=4, figure=fig, height_ratios=[1.0, 1.35], width_ratios=[1, 1, 1, 0.10])

map_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
cax_map = fig.add_subplot(gs[0, 3])

cmap_map = "RdBu_r"
cmap_map_obj = plt.get_cmap(cmap_map)
norm_map = Normalize(vmin=-vmax_pct, vmax=vmax_pct)

for ax, ev in zip(map_axes, events_for_fig):
    g_ev = attach_event_to_gdf(ev)

    g_ev.plot(
        ax=ax, column="Delta_pct",
        cmap=cmap_map, norm=norm_map,
        linewidth=1.0, edgecolor="white"
    )
    state_outline.plot(ax=ax, color="black", linewidth=1.1)

    g_sig = g_ev[g_ev["p_value"] < 0.05]
    if len(g_sig):
        g_sig.boundary.plot(ax=ax, color="black", linewidth=2.0)
    g_marg = g_ev[(g_ev["p_value"] >= 0.05) & (g_ev["p_value"] < 0.10)]
    if len(g_marg):
        g_marg.boundary.plot(ax=ax, color="#555555", linewidth=1.6)

    for _, r in g_ev.iterrows():
        if r["centroid"] is None or r["centroid"].is_empty:
            continue
        tcol = text_color_for_bg(r["Delta_pct"], norm_map, cmap_map_obj, light_thresh=0.55)
        stroke_col = "black" if tcol == "white" else "white"

        ax.text(
            r["centroid"].x, r["centroid"].y, r["abbr"],
            ha="center", va="center",
            fontsize=14,
            fontweight="bold",
            color=tcol,
            alpha=0.98,
            path_effects=[
                pe.withStroke(linewidth=1.6, foreground=stroke_col),
                pe.Normal()
            ]
        )

    ax.set_title(f"{ev}", pad=6)
    ax.set_axis_off()

sm_map = ScalarMappable(norm=norm_map, cmap=cmap_map)
sm_map.set_array([])
cb_map = fig.colorbar(sm_map, cax=cax_map, orientation="vertical")
cb_map.set_label("Δ% consumption", labelpad=8)

ax_bub = fig.add_subplot(gs[1, :])

mat = (comp.pivot(index="Region", columns="Event", values="Delta_z")
           .reindex(index=regions_order, columns=events_for_fig))
pmat = (comp.pivot(index="Region", columns="Event", values="p_value")
            .reindex(index=regions_order, columns=events_for_fig))

xnames = events_for_fig
xpos = np.arange(len(xnames))

y_spacing = 1.15
ypos = (np.arange(len(regions_order))[::-1]) * y_spacing
region_to_y = {r: ypos[i] for i, r in enumerate(regions_order)}

max_abs = np.nanmax(np.abs(mat.values)) if np.isfinite(mat.values).any() else 1.0
max_abs = max(0.25, float(max_abs))
size_min, size_max = 50, 900

cmap_bub = "RdBu_r"
norm_bub = Normalize(vmin=-vmax_z, vmax=vmax_z)

for r in regions_order:
    for j, ev in enumerate(xnames):
        val = mat.loc[r, ev]
        p = pmat.loc[r, ev]
        if pd.isna(val):
            continue

        s = size_min + (np.abs(val) / max_abs) ** 1.35 * (size_max - size_min)

        if pd.notna(p) and p < 0.05:
            ec, lw = "black", 1.6
        elif pd.notna(p) and p < 0.10:
            ec, lw = "#444444", 1.2
        else:
            ec, lw = "none", 0.0

        ax_bub.scatter(
            j, region_to_y[r],
            s=s, c=[val],
            cmap=cmap_bub, norm=norm_bub,
            edgecolors=ec, linewidths=lw,
            alpha=0.95
        )

ax_bub.set_yticks(list(region_to_y.values()))
ax_bub.set_yticklabels(regions_order)
ax_bub.set_xticks(xpos)

ax_bub.set_xticklabels([ev for ev in xnames])

ax_bub.set_xlim(-0.6, len(xnames) - 0.4)

y_min = -1.0 * y_spacing
y_max = (len(regions_order) - 1) * y_spacing + 1.0 * y_spacing
ax_bub.set_ylim(y_min, y_max)

ax_bub.grid(True, axis="both", alpha=0.15, linewidth=0.8)
ax_bub.spines["top"].set_visible(False)
ax_bub.spines["right"].set_visible(False)

cax2 = ax_bub.inset_axes([1.01, 0.10, 0.02, 0.80])
sm_bub = ScalarMappable(norm=norm_bub, cmap=cmap_bub)
sm_bub.set_array([])
cb_bub = fig.colorbar(sm_bub, cax=cax2, orientation="vertical")
cb_bub.set_label("Δz consumption", labelpad=10)

legend_vals = [0.25, 0.50, 1.00]
legend_vals = [v for v in legend_vals if v <= max_abs + 1e-9]
handles = []
for v in legend_vals:
    s = size_min + (np.abs(v) / max_abs) ** 1.35 * (size_max - size_min)
    h = ax_bub.scatter([], [], s=s, c="none", edgecolors="#333333", linewidths=1.0)
    handles.append(h)

ax_bub.legend(
    handles, [f"|Δz| = {v:.2f}" for v in legend_vals],
    title="Effect size",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.01, 0.99),
    labelspacing=1.6,
    handletextpad=0.8,
    borderaxespad=0.2
)

plt.tight_layout(rect=[0.0, 0.0, 0.98, 0.98])

plt.savefig(os.path.join(OUT_DIR, "figure 4 - event composites.jpg"), dpi=600, bbox_inches="tight")
plt.close()
