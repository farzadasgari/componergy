from componergy.paths import COUNTY_CONSUMPTION_XLSX, CALIFORNIA_MONTHLY_FILE
from componergy.paths import FIGURES_DIR
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import json
from urllib.request import urlopen
from shapely.geometry import shape
from shapely import wkt
from pathlib import Path
import zipfile
import io
import requests
import os

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = str(FIGURES_DIR / "Figure 3")
os.makedirs(OUT_DIR, exist_ok=True)

FIG_HEATMAP = os.path.join(OUT_DIR, "figure 3 - regional - heatmaps.jpg")
FIG_SLOPE = os.path.join(OUT_DIR, "figure 3 - regional - slope.jpg")

CSV_DOM = os.path.join(OUT_DIR, "figure 3 - driver dominance.csv")
CSV_STATS = os.path.join(OUT_DIR, "figure 3 - driver overall stats.csv")
CSV_LAG = os.path.join(OUT_DIR, "figure 3 - lag summary.csv")
CSV_SIGN = os.path.join(OUT_DIR, "figure 3 - sign consistency.csv")
CSV_INCR = os.path.join(OUT_DIR, "figure 3 - incremental compound value.csv")

REGION_RENAME = {
    "Sonora Desert": "Sonoran Desert",
}

LOG_ALL = os.path.join(OUT_DIR, "figure 3 - regional.log")

TMP_COUNTIES_DIR = os.path.join(OUT_DIR, "_tmp_counties")

COLOR_STI = "crimson"
COLOR_SAPEI = "deepskyblue"
COLOR_SCDHI = "purple"
COLOR_ELECTRICITY = "#222222"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "axes.linewidth": 1.2,
    "savefig.dpi": 600,
})

consumption_path = str(COUNTY_CONSUMPTION_XLSX)
netcdf_path = str(CALIFORNIA_MONTHLY_FILE)
regions_url = "https://api.cal-adapt.org/api/climregions/"


def zscore_frame(df):
    return (df - df.mean(axis=0)) / df.std(axis=0, ddof=0)


def fetch_all_caladapt(url, pagesize=2000):
    sep = "&" if "?" in url else "?"
    next_url = f"{url}{sep}pagesize={pagesize}"
    feats = []
    meta = None
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
    tmp = Path(TMP_COUNTIES_DIR)
    tmp.mkdir(exist_ok=True)
    zf.extractall(tmp)
    shp = next(tmp.glob("*.shp"))
    gdf = gpd.read_file(shp)
    gdf = gdf[gdf["STATEFP"] == "06"].to_crs("EPSG:4326")
    gdf["CountyName"] = gdf["NAME"].astype("string").str.upper().str.strip()
    return gdf[["CountyName", "geometry"]].copy()


def build_county_to_region_map(counties_gdf, regions_gdf):
    counties_centroids = counties_gdf.copy()
    counties_centroids["geometry"] = counties_centroids.geometry.representative_point()
    joined = gpd.sjoin(counties_centroids, regions_gdf[["Region", "geometry"]],
                       how="left", predicate="within")
    mapping = joined[["CountyName", "Region"]].drop_duplicates()
    unmapped = mapping[mapping["Region"].isna(
    )]["CountyName"].unique().tolist()
    if len(unmapped) > 0:
        raise ValueError(
            f"Unmapped counties (check geometry/join): {unmapped[:10]}{'...' if len(unmapped) > 10 else ''}")
    return mapping


def to_time_index_from_year_month(da):
    da_1d = da.stack(time=("year", "month")).sortby("time")
    y = da_1d["year"].values
    m = da_1d["month"].values
    idx = pd.to_datetime(
        [f"{int(yy):04d}-{int(mm):02d}-01" for yy, mm in zip(y, m)])
    return da_1d, idx


def regional_climate_timeseries(ds_window, regions_gdf, varname):
    import regionmask

    lat = ds_window["lat"]
    lon = ds_window["lon"]

    regions_gdf = regions_gdf.copy()
    regions_gdf["geometry"] = regions_gdf["geometry"].buffer(0)

    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat = weights_lat / weights_lat.mean()

    regions = regionmask.Regions(
        outlines=list(regions_gdf.geometry),
        names=list(regions_gdf["Region"]),
        abbrevs=[f"R{i+1:02d}" for i in range(len(regions_gdf))]
    )

    try:
        mask_2d = regions.mask(lon, lat)
    except ValueError:
        mask_3d = regions.mask_3D(lon, lat)
        any_region = mask_3d.any("region")
        first_region = mask_3d.argmax("region")
        mask_2d = first_region.where(any_region)

    out = {}
    for i, name in enumerate(regions.names):
        da = ds_window[varname].where(mask_2d == i)
        da_mean = da.weighted(weights_lat).mean(dim=["lat", "lon"])
        da_1d, idx = to_time_index_from_year_month(da_mean)
        out[name] = pd.Series(da_1d.values, index=idx)

    return pd.DataFrame(out).sort_index()


df = pd.read_excel(consumption_path, sheet_name="Electricity Consumption")
df["Date"] = pd.to_datetime(df["Year"].astype(
    str) + "-" + df["Month"].astype(str) + "-01")
df["CountyName"] = df["CountyName"].astype("string").str.upper().str.strip()

regions_gdf = load_climate_regions(regions_url)
counties_gdf = download_ca_counties_cartographic()
county_region = build_county_to_region_map(counties_gdf, regions_gdf)

df = df.merge(county_region, on="CountyName", how="left")
if df["Region"].isna().any():
    missing = df.loc[df["Region"].isna(
    ), "CountyName"].dropna().unique().tolist()
    raise ValueError(
        f"Counties missing region assignment: {missing[:10]}{'...' if len(missing) > 10 else ''}")

electricity_regional = (
    df.groupby(["Date", "Region"])["Consumption (MWh)"]
      .sum()
      .unstack("Region")
      .sort_index()
    / 1000.0
)

ds = xr.open_dataset(netcdf_path)
ds_window = ds.sel(year=slice(2008, 2024))

sti_regional = regional_climate_timeseries(ds_window, regions_gdf, "sti_scdhi")
sapei_regional = regional_climate_timeseries(
    ds_window, regions_gdf, "sapei_scdhi")
scdhi_regional = regional_climate_timeseries(ds_window, regions_gdf, "scdhi")

start = max(electricity_regional.index.min(), sti_regional.index.min(),
            sapei_regional.index.min(), scdhi_regional.index.min())
end = min(electricity_regional.index.max(), sti_regional.index.max(),
          sapei_regional.index.max(), scdhi_regional.index.max())

electricity_regional = electricity_regional.loc[start:end]
sti_regional = sti_regional.loc[start:end]
sapei_regional = sapei_regional.loc[start:end]
scdhi_regional = scdhi_regional.loc[start:end]

smooth_window = 3
electricity_regional = electricity_regional.rolling(
    smooth_window, center=True, min_periods=1).mean()
sti_regional = sti_regional.rolling(
    smooth_window, center=True, min_periods=1).mean()
sapei_regional = sapei_regional.rolling(
    smooth_window, center=True, min_periods=1).mean()
scdhi_regional = scdhi_regional.rolling(
    smooth_window, center=True, min_periods=1).mean()

electricity_z = zscore_frame(electricity_regional)
sti_z = zscore_frame(sti_regional)
sapei_z = zscore_frame(sapei_regional)
scdhi_z = zscore_frame(scdhi_regional)

regions_order = list(regions_gdf["Region"])
regions_order = [r for r in regions_order if r in electricity_z.columns]


# def plot_heatmap(ax, data, title, vlim=2.5, cmap="viridis"):
#     mat = data[regions_order].T.values
#     im = ax.imshow(mat, aspect="auto", interpolation="nearest",
#                    vmin=-vlim, vmax=vlim, cmap=cmap)
#     ax.set_title(title, pad=6)
#     ax.set_yticks(np.arange(len(regions_order)))
#     ax.set_yticklabels(regions_order)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     return im


def plot_heatmap(ax, data, panel_letter=None, row_label=None, vlim=2.5, cmap="viridis"):
    mat = data[regions_order].T.values
    im = ax.imshow(
        mat, aspect="auto", interpolation="nearest",
        vmin=-vlim, vmax=vlim, cmap=cmap
    )

    # No title (journal-style)
    ax.set_yticks(np.arange(len(regions_order)))
    ax.set_yticklabels(regions_order)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel letter (a, b, c, d)
    # if panel_letter is not None:
    #     ax.text(
    #         -0.10, 1.02, panel_letter,
    #         transform=ax.transAxes,
    #         ha="left", va="bottom",
    #         fontsize=14, fontweight="bold"
    #     )

    # Row label on the right side (cleaner than titles)
    if row_label is not None:
        ax.text(
            1.02, 0.5, row_label,
            transform=ax.transAxes,
            rotation=90,
            ha="center", va="center",
            fontsize=16
        )

    return im

# fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10),
#                          sharex=False, constrained_layout=True)

# im0 = plot_heatmap(axes[0], electricity_z, "Regional electricity consumption anomaly (z-score)")
# plot_heatmap(axes[1], sti_z, "Regional STI anomaly (z-score)")
# plot_heatmap(axes[2], sapei_z, "Regional SAPEI anomaly (z-score)")
# plot_heatmap(axes[3], scdhi_z, "Regional SCDHI anomaly (z-score)")

# for ax in axes[0:3]:
#     ax.set_xticklabels([])
# axes[-1].set_xlabel("Time (Year)")

# tick_idx = np.arange(0, len(electricity_z.index), 24)
# tick_labels = [electricity_z.index[i].strftime("%Y") for i in tick_idx]
# for ax in axes:
#     ax.set_xticks(tick_idx)
#     ax.set_xticklabels(tick_labels)

# cbar = fig.colorbar(
#     im0,
#     ax=axes,
#     orientation="vertical",
#     fraction=0.030,
#     pad=0.02,
#     shrink=0.92,
#     aspect=35
# )
# cbar.set_label("Standardized anomaly (z-score)", fontsize=16)
# cbar.ax.tick_params(labelsize=18)
# cbar.set_label("Standardized anomaly (z-score)")

# plt.savefig(FIG_HEATMAP, bbox_inches="tight")
# plt.close()


fig, axes = plt.subplots(
    nrows=4, ncols=1,
    figsize=(12, 12),
    sharex=True,
    constrained_layout=True
)
# Put climate drivers first, electricity last
plot_heatmap(axes[0], sti_z,    panel_letter="a", row_label="STI")
plot_heatmap(axes[1], sapei_z,  panel_letter="b", row_label="SAPEI")
plot_heatmap(axes[2], scdhi_z,  panel_letter="c", row_label="SCDHI")
im0 = plot_heatmap(axes[3], electricity_z,
                   panel_letter="d", row_label="Electricity")

# Hide x tick labels on upper panels (cleaner than set_xticklabels([]))
for ax in axes[:-1]:
    ax.tick_params(axis="x", labelbottom=False)

axes[-1].set_xlabel("Time (Year)")

tick_idx = np.arange(0, len(electricity_z.index), 24)
tick_labels = [electricity_z.index[i].strftime("%Y") for i in tick_idx]
for ax in axes:
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels)

cbar = fig.colorbar(
    im0,
    ax=axes,
    orientation="vertical",
    fraction=0.030,
    pad=0.02,
    shrink=0.92,
    aspect=35
)
cbar.set_label("Standardized anomaly", fontsize=16)
cbar.ax.tick_params(labelsize=18)

plt.savefig(FIG_HEATMAP, bbox_inches="tight")
plt.close()

lags = np.arange(-6, 7)
regions_ok = regions_order


def best_lag_slope(x, y):
    best = None
    for L in lags:
        if L < 0:
            xL = x.iloc[-L:].values
            yL = y.iloc[:len(xL)].values
        elif L > 0:
            xL = x.iloc[:-L].values
            yL = y.iloc[L:].values
        else:
            xL = x.values
            yL = y.values
        if len(xL) < 24:
            continue
        A = np.vstack([xL, np.ones_like(xL)]).T
        slope, intercept = np.linalg.lstsq(A, yL, rcond=None)[0]
        yhat = slope * xL + intercept
        ssr = np.sum((yL - yhat) ** 2)
        sst = np.sum((yL - np.mean(yL)) ** 2)
        r2 = 1.0 - ssr / sst if sst > 0 else np.nan
        cand = (r2, L, slope)
        if best is None or (np.nan_to_num(cand[0], nan=-np.inf) > np.nan_to_num(best[0], nan=-np.inf)):
            best = cand
    return best


rows = []
for region in regions_ok:
    y = electricity_z[region].dropna()
    for name, X in [("STI", sti_z[region]), ("SAPEI", sapei_z[region]), ("SCDHI", scdhi_z[region])]:
        x = X.reindex(y.index).dropna()
        yy = y.reindex(x.index).dropna()
        xx = x.reindex(yy.index)
        r2, lag, slope = best_lag_slope(xx, yy)
        rows.append({
            "Region": region,
            "Driver": name,
            "BestLagMonths": lag,
            "Slope_Electricity_per_z": slope,
            "R2": r2
        })

summary = pd.DataFrame(rows)
summary = summary.sort_values(["Driver", "R2"], ascending=[True, False])

fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
drivers = ["STI", "SAPEI", "SCDHI"]
driver_colors = {"STI": COLOR_STI, "SAPEI": COLOR_SAPEI, "SCDHI": COLOR_SCDHI}
xpos = np.arange(len(regions_ok))
width = 0.26

for j, drv in enumerate(drivers):
    sub = summary[summary["Driver"] == drv].set_index("Region").loc[regions_ok]
    ax.bar(
        xpos + (j - 1) * width,
        sub["Slope_Electricity_per_z"].values,
        width=width,
        label=drv,
        color=driver_colors[drv]
    )

ax.axhline(0, linewidth=1.0, color="#444444", alpha=0.6)
ax.set_xticks(xpos)
ax.set_xticklabels(regions_ok, rotation=90, ha="right")
ax.set_ylabel("Electricity Response")
ax.legend(frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(0.5, 1.1))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig(FIG_SLOPE, bbox_inches="tight")
plt.close()


dominance = (
    summary.sort_values("R2", ascending=False)
           .groupby("Region")
           .first()
           .reset_index()[["Region", "Driver", "R2", "BestLagMonths", "Slope_Electricity_per_z"]]
)
dominance.to_csv(CSV_DOM, index=False)

driver_stats = (
    summary.groupby("Driver")
           .agg(
               mean_slope=("Slope_Electricity_per_z", "mean"),
               median_slope=("Slope_Electricity_per_z", "median"),
               std_slope=("Slope_Electricity_per_z", "std"),
               mean_R2=("R2", "mean"),
               median_R2=("R2", "median")
    )
    .reset_index()
)
driver_stats.to_csv(CSV_STATS, index=False)

lag_summary = (
    summary.groupby("Driver")
           .agg(
               mean_lag=("BestLagMonths", "mean"),
               median_lag=("BestLagMonths", "median"),
               min_lag=("BestLagMonths", "min"),
               max_lag=("BestLagMonths", "max")
    )
    .reset_index()
)
lag_summary.to_csv(CSV_LAG, index=False)

sign_check = (
    summary.assign(sign=lambda df_: np.sign(df_["Slope_Electricity_per_z"]))
    .groupby("Driver")["sign"]
    .value_counts()
    .unstack(fill_value=0)
)
sign_check.to_csv(CSV_SIGN)


def r2_simple(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope * x + intercept
    ssr = np.sum((y - yhat) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    return 1 - ssr / sst if sst > 0 else np.nan


incremental_rows = []
for region in regions_ok:
    y = electricity_z[region].dropna()
    sti = sti_z[region].reindex(y.index).dropna()
    scd = scdhi_z[region].reindex(y.index).dropna()

    dfR = pd.concat([y.rename("y"), sti.rename("sti"),
                    scd.rename("scd")], axis=1).dropna()
    yv = dfR["y"].values
    sti_v = dfR["sti"].values
    scd_v = dfR["scd"].values

    r2_sti = r2_simple(sti_v, yv)

    X = np.vstack([sti_v, scd_v, np.ones_like(sti_v)]).T
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    yhat = X @ beta
    ssr = np.sum((yv - yhat) ** 2)
    sst = np.sum((yv - np.mean(yv)) ** 2)
    r2_both = 1 - ssr / sst if sst > 0 else np.nan

    incremental_rows.append({
        "Region": region,
        "R2_STI_only": r2_sti,
        "R2_STI_plus_SCDHI": r2_both,
        "Delta_R2": r2_both - r2_sti
    })

incremental = pd.DataFrame(incremental_rows)
incremental.to_csv(CSV_INCR, index=False)

log_lines = []
log_lines.append("=== Figure 3 summary ===")
log_lines.append("")
log_lines.append(f"Period: {start.date()} to {end.date()}")
log_lines.append(f"Smoothing: {smooth_window}-month centered rolling mean")
log_lines.append(f"Regions: {len(regions_ok)}")
log_lines.append("")

log_lines.append("=== Dominant driver per region (highest R2) ===")
log_lines.append(dominance.to_string(index=False))
log_lines.append("")

log_lines.append("=== Driver overall statistics across regions ===")
log_lines.append(driver_stats.to_string(index=False))
log_lines.append("")

log_lines.append("=== Lag summary per driver ===")
log_lines.append(lag_summary.to_string(index=False))
log_lines.append("")

log_lines.append("=== Sign consistency of slopes ===")
log_lines.append(sign_check.to_string())
log_lines.append("")

log_lines.append("=== Outputs ===")
log_lines.append(f"Heatmaps: {FIG_HEATMAP}")
log_lines.append(f"Slope plot: {FIG_SLOPE}")
log_lines.append(f"CSV dominance: {CSV_DOM}")
log_lines.append(f"CSV driver stats: {CSV_STATS}")
log_lines.append(f"CSV lag summary: {CSV_LAG}")
log_lines.append(f"CSV sign consistency: {CSV_SIGN}")
log_lines.append(f"CSV incremental: {CSV_INCR}")
log_lines.append("")

with open(LOG_ALL, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
