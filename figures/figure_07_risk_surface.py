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

OUT_DIR = str(FIGURES_DIR / "Figure 7")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
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

    gdf["Region"] = gdf[name_col].astype("string").str.strip()
    gdf = gdf[["Region", "geometry"]].dropna().drop_duplicates(subset=["Region"]).reset_index(drop=True)
    gdf["geometry"] = gdf["geometry"].buffer(0)
    return gdf


def download_ca_counties_cartographic():
    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
    content = requests.get(url, timeout=120).content
    zf = zipfile.ZipFile(io.BytesIO(content))
    tmp = Path(f"{OUT_DIR}/_tmp_counties")
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
    lat = ds_window["lat"]
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    da = ds_window[varname].weighted(w).mean(dim=["lat", "lon"])
    da_1d, idx = to_time_index_from_year_month(da)
    return pd.Series(da_1d.values, index=idx).sort_index()


def region_abbrev(name: str) -> str:
    manual = {
        "San Joaquin Valley": "SJV",
        "Sacramento-Delta": "SD",
        "Central Coast": "CC",
        "North Coast": "NC",
        "South Coast": "SC",
        "North Central": "N-C",
        "Southern Interior": "SI",
        "Mojave Desert": "MD",
        "Sonora Desert": "SoD",
        "Northeast": "NE",
        "Sierra": "SR",
    }
    if name in manual:
        return manual[name]
    parts = [p for p in name.replace("-", " ").split() if p]
    return "".join(p[0].upper() for p in parts[:3])


def text_color_for_bg(value, norm, cmap_obj, light_thresh=0.55):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "black"
    r, g, b, _ = cmap_obj(norm(value))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if lum >= light_thresh else "white"


def add_letter(ax, letter, x=-0.08, y=0.96, fs=26):
    """Bigger letters, normal weight, moved LEFT outside the map frames"""
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=fs, fontweight="normal", color="#222222")


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
    missing = df.loc[df["Region"].isna()], "CountyName".dropna().unique().tolist()
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

E_state = electricity_regional.sum(axis=1)
E_state_local = deseasonalize_monthly(E_state.to_frame(name="Demand_GWh"))["Demand_GWh"]
E_state_z = zscore_frame(E_state_local.to_frame(name="z"))["z"]

gen_stress = -SCDHI
demand_surge = E_state_z
monthly_risk = gen_stress * demand_surge

evi = comp.pivot(index="Region", columns="Event", values="Delta_z").mean(axis=1).rename("EVI")

ds_stacked = ds_window.stack(time=("year", "month")).sortby("time")
y = ds_stacked["year"].values
m = ds_stacked["month"].values
time_idx = pd.to_datetime([f"{int(yy):04d}-{int(mm):02d}-01" for yy, mm in zip(y, m)])
ds_stacked = ds_stacked.assign_coords(time=("time", time_idx))

common_times = pd.Index(time_idx).intersection(event_defs["Heatwave"].index)

mask_heat = event_defs["Heatwave"].reindex(common_times).fillna(False).values
mask_drought = event_defs["Drought"].reindex(common_times).fillna(False).values
mask_comp = event_defs["Compound"].reindex(common_times).fillna(False).values
mask_normal = normal.reindex(common_times).fillna(False).values

def mean_over_mask(da, mask):
    if mask.sum() == 0:
        return da.mean(dim="time") * np.nan
    return da.where(mask, drop=False).mean(dim="time")

sti_ev = mean_over_mask(ds_stacked["sti_scdhi"].sel(time=common_times), mask_heat)
sti_nm = mean_over_mask(ds_stacked["sti_scdhi"].sel(time=common_times), mask_normal)
delta_heat = sti_ev - sti_nm

sapei_ev = mean_over_mask(ds_stacked["sapei_scdhi"].sel(time=common_times), mask_drought)
sapei_nm = mean_over_mask(ds_stacked["sapei_scdhi"].sel(time=common_times), mask_normal)
delta_drought = -(sapei_ev - sapei_nm)

scdhi_ev = mean_over_mask(ds_stacked["scdhi"].sel(time=common_times), mask_comp)
scdhi_nm = mean_over_mask(ds_stacked["scdhi"].sel(time=common_times), mask_normal)
delta_comp = -(scdhi_ev - scdhi_nm)

gridded_evi = (delta_heat + delta_drought + delta_comp) / 3

risk_values = np.sort(monthly_risk.dropna().values)[::-1]
n = len(risk_values)
ranks = np.arange(1, n + 1)
return_period_months = (n + 1.0) / ranks
return_period_years = return_period_months / 12.0

risk_series = monthly_risk.dropna()
top_extremes = pd.DataFrame({
    "Date": risk_series.sort_values(ascending=False).index[:10],
    "System_Risk": risk_values[:10],
    "Return_Period_Years": return_period_years[:10]
})

evi.to_frame().to_csv(os.path.join(OUT_DIR, "figure 7 - regional_evi.csv"))
pd.DataFrame({
    "Date": monthly_risk.index,
    "Gen_Stress": gen_stress,
    "Demand_Surge": demand_surge,
    "System_Risk": monthly_risk
}).to_csv(os.path.join(OUT_DIR, "figure 7 - monthly_system_risk.csv"), index=False)
top_extremes.to_csv(os.path.join(OUT_DIR, "figure 7 - top_extremes.csv"), index=False)

lons = ds_window["lon"].values
lats = ds_window["lat"].values
Lon, Lat = np.meshgrid(lons, lats)
pd.DataFrame({
    "lat": Lat.flatten(),
    "lon": Lon.flatten(),
    "gridded_evi": gridded_evi.values.flatten()
}).to_csv(os.path.join(OUT_DIR, "figure 7 - gridded_evi.csv"), index=False)

with open(os.path.join(OUT_DIR, "figure 7 - systemic_vulnerability.log"), "w", encoding="utf-8") as f:
    f.write("=== Figure 7: Systemic Vulnerability Metric (2×2 Nature-style) ===\n\n")
    f.write(f"Period: {start.date()} to {end.date()}  | months={len(E.index)}\n")
    f.write(f"Smoothing: {smooth_window}-month centered rolling mean\n\n")
    f.write("Event thresholds (statewide):\n")
    f.write(f"  Heat: STI ≥ q{q_heat:.2f} (thr={thr_heat:.3f})\n")
    f.write(f"  Drought: SAPEI ≤ q{q_drought:.2f} (thr={thr_drought:.3f})\n")
    f.write(f"  Compound: SCDHI ≤ q{q_compound:.2f} (thr={thr_comp:.3f})\n")
    f.write(f"  Normal: SCDHI ≥ q{q_normal:.2f} (thr={thr_norm:.3f})\n\n")
    f.write(f"Counts: Normal={N_normal} | " + " | ".join([f"{k}={v}" for k, v in N_events.items()]) + "\n\n")
    f.write("=== PANEL DEFINITIONS ===\n")
    f.write("• A: Statewide gridded cells — composite climate stress Δ (STI heat / –SAPEI drought / –SCDHI compound) averaged across events\n")
    f.write("• B: Zonal (regional) EVI — mean demand Δz (event – normal) across Heatwave/Drought/Compound\n")
    f.write("• C: Risk surface (generation stress × demand surge)\n")
    f.write("• D: Return period of system stress events\n\n")
    f.write("=== KEY NUMBERS FOR MANUSCRIPT ===\n")
    f.write(f"Max System Risk: {monthly_risk.max():.3f} (on {monthly_risk.idxmax().date()})\n")
    f.write(f"Return period of most extreme event: {return_period_years[0]:.1f} years\n")
    f.write(f"Months with positive risk: {(monthly_risk > 0).sum()}/{len(monthly_risk)} ({100*(monthly_risk>0).mean():.1f}%)\n")
    f.write(f"90th-percentile System Risk: {np.nanpercentile(monthly_risk, 90):.3f}\n")
    f.write(f"Regional EVI range: {evi.min():.3f} – {evi.max():.3f}\n")
    f.write(f"Highest EVI region: {evi.idxmax()} ({evi.max():.3f})\n")
    f.write(f"Gridded EVI range: {float(gridded_evi.min()):.3f} – {float(gridded_evi.max()):.3f}\n")
    f.write("\nPanel A = raw climate hazard map | Panel B = observed demand impact map. Perfect complementarity!\n")

state_outline = counties_gdf.dissolve().boundary
regions_plot = regions_gdf.copy()
regions_plot["abbr"] = regions_plot["Region"].map(region_abbrev)
regions_plot["centroid"] = regions_plot.geometry.representative_point()

g_evi = regions_plot.copy()
g_evi["EVI"] = g_evi["Region"].map(evi.to_dict())

delta_z_all = comp["Delta_z"].dropna().values
grid_abs = np.abs(gridded_evi.values.flatten())
grid_abs = grid_abs[np.isfinite(grid_abs)]
vmax_evi = max(0.5,
               float(np.nanpercentile(np.abs(delta_z_all), 90)),
               float(np.nanpercentile(grid_abs, 90)) if len(grid_abs) > 0 else 0.5)

fig = plt.figure(figsize=(16.2, 11.8), constrained_layout=False)
gs = GridSpec(nrows=2, ncols=2, figure=fig,
              width_ratios=[1.08, 1.0], height_ratios=[1.05, 1.0],
              wspace=0.4, hspace=0.12)
fig.subplots_adjust(left=0.04, right=0.90, top=0.96, bottom=0.07)

cmap_map = "RdBu_r"
norm_map = Normalize(vmin=-vmax_evi, vmax=vmax_evi)
cmap_obj = plt.get_cmap(cmap_map)

axA = fig.add_subplot(gs[0, 0])
lons = ds_window["lon"].values
lats = ds_window["lat"].values
XX, YY = np.meshgrid(lons, lats)

axA.pcolormesh(XX, YY, gridded_evi.values,
               cmap=cmap_map, norm=norm_map, shading="auto", rasterized=True)

state_outline.plot(ax=axA, color="black", linewidth=1.2)
regions_gdf.plot(ax=axA, facecolor="none", edgecolor="#333333", linewidth=0.75, alpha=0.45)

axA.set_axis_off()
add_letter(axA, "A", x=-0.08)
cax_A = fig.add_axes([0.455, 0.57, 0.013, 0.36])

sm_A = ScalarMappable(
    norm=Normalize(
        vmin=float(gridded_evi.min()),
        vmax=float(gridded_evi.max())
    ),
    cmap=cmap_map
)

cb_A = fig.colorbar(sm_A, cax=cax_A)
cb_A.set_label("Climate Vulnerability")

axB = fig.add_subplot(gs[0, 1])
g_evi.plot(ax=axB, column="EVI", cmap=cmap_map, norm=norm_map,
           linewidth=1.0, edgecolor="white")
state_outline.plot(ax=axB, color="black", linewidth=1.2)

for _, r in g_evi.iterrows():
    if r["centroid"] is None or r["centroid"].is_empty:
        continue
    tcol = text_color_for_bg(r["EVI"], norm_map, cmap_obj, light_thresh=0.55)
    stroke_col = "black" if tcol == "white" else "white"
    axB.text(
        r["centroid"].x, r["centroid"].y, r["abbr"],
        ha="center", va="center", fontsize=16, fontweight="bold",
        color=tcol, alpha=0.98,
        path_effects=[pe.withStroke(linewidth=1.6, foreground=stroke_col), pe.Normal()]
    )

axB.set_axis_off()
add_letter(axB, "B", x=-0.08)

cax_B = fig.add_axes([0.905, 0.57, 0.013, 0.36])

sm_B = ScalarMappable(
    norm=Normalize(
        vmin=evi.min(),
        vmax=evi.max()
    ),
    cmap=cmap_map
)

cb_B = fig.colorbar(sm_B, cax=cax_B)
cb_B.set_label("Demand Vulnerability")

axC = fig.add_subplot(gs[1, 0])
axC.scatter(gen_stress, demand_surge,
            c=monthly_risk, cmap="RdBu_r",
            norm=Normalize(vmin=monthly_risk.min(), vmax=monthly_risk.max()),
            s=70, alpha=0.88, edgecolors="none")

x_range = np.linspace(gen_stress.min() - 0.2, gen_stress.max() + 0.2, 90)
y_range = np.linspace(demand_surge.min() - 0.2, demand_surge.max() + 0.2, 90)
Xg, Yg = np.meshgrid(x_range, y_range)
Zg = Xg * Yg
risk_levels = np.linspace(0.5, monthly_risk.max() * 0.85, 6)
risk_levels = risk_levels[risk_levels > 0.2]
if len(risk_levels) > 0:
    cs = axC.contour(Xg, Yg, Zg, levels=risk_levels, colors="#222222",
                     linewidths=2, linestyles="dashed", alpha=0.85)
    axC.clabel(cs, inline=True, fontsize=14, fmt="Risk=%.1f", colors="#222222")

axC.axvline(0, color="black", lw=1.1, alpha=0.45)
axC.axhline(0, color="black", lw=1.1, alpha=0.45)

axC.set_xlabel("Generation stress (−SCDHI)")
axC.set_ylabel("Demand surge (z-score)")
add_letter(axC, "C", x=0.02)

cax_risk = axC.inset_axes([1.07, 0.08, 0.028, 0.75])
sm_risk = ScalarMappable(norm=Normalize(monthly_risk.min(), monthly_risk.max()), cmap="RdBu_r")
fig.colorbar(sm_risk, cax=cax_risk).set_label("System Risk", labelpad=8)

axD = fig.add_subplot(gs[1, 1])
axD.plot(return_period_years, risk_values, "o-", color="#AF0A0A",
         markersize=8, linewidth=3.4, alpha=0.92)

axD.set_xlabel("Return period (years)")
axD.set_ylabel("System risk index")
axD.set_xscale("log")
add_letter(axD, "D", x=0.02)

axD.set_xticks([1, 2, 5, 10, 20])
axD.set_xticklabels(["1", "2", "5", "10", "20"])

max_rp = return_period_years[0]
axD.annotate(f"Observed max\nRP = {max_rp:.1f} yr",
             xy=(max_rp, risk_values[0]),
             xytext=(max_rp * 0.65, risk_values[0] * 1.18),
             fontsize=17, color="#8B0000",
             arrowprops=dict(arrowstyle="->", color="#8B0000", lw=1.9))

axD.grid(True, alpha=0.15)

plt.savefig(os.path.join(OUT_DIR, "figure 7 - systemic vulnerability.jpg"), bbox_inches="tight", dpi=600)
plt.close()
