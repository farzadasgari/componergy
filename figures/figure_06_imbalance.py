from componergy.paths import CALIFORNIA_MONTHLY_FILE, GENERATION_MONTHLY_FILE, DEMAND_MONTHLY_FILE, FIGURES_DIR
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, ttest_ind
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NC_PATH = str(CALIFORNIA_MONTHLY_FILE)
GEN_PATH = str(GENERATION_MONTHLY_FILE)
DEM_PATH = str(DEMAND_MONTHLY_FILE)

OUT_DIR = str(FIGURES_DIR / "Figure 6")
os.makedirs(OUT_DIR, exist_ok=True)

FIG_PATH = os.path.join(OUT_DIR, "figure 6 - supply demand imbalance.jpg")
CSV_METRICS = os.path.join(OUT_DIR, "figure 6 - imbalance_metrics.csv")
CSV_EVENTS = os.path.join(OUT_DIR, "figure 6 - event_metrics.csv")
CSV_BOOT = os.path.join(OUT_DIR, "figure 6 - bootstrap_raf.csv")
LOG_PATH = os.path.join(OUT_DIR, "figure 6 - supply demand imbalance.log")

SMOOTH_MONTHS = 3
P_HEAT = 0.90
P_DROUGHT = 0.10
P_COMP = 0.10
P_NORMAL = 0.90
RISK_Q = 0.90
N_BOOT = 5000
BOOT_SEED = 42
EPS = 1e-9

C_HEAT = "crimson"
C_DROUGHT = "deepskyblue"
C_COMP = "purple"
C_IMB = "#222222"
C_GEN = "#d95f02"
C_DEM = "#1f78b4"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "axes.linewidth": 1.4,
    "savefig.dpi": 600,
})


def add_letter(ax, letter, x=0.01, y=0.98, fs=18):
    ax.text(x, y, letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=fs)


def zscore(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    sd = s.std(ddof=0)
    if sd and np.isfinite(sd):
        return (s - s.mean()) / sd
    return s * 0.0


def rolling_mean(s, w):
    return s.rolling(w, center=True, min_periods=1).mean()


def guess_lat_lon_names(ds):
    lat_candidates = ["lat", "latitude", "y", "YLAT", "LAT"]
    lon_candidates = ["lon", "longitude", "x", "XLONG", "LON"]
    lat = next((n for n in lat_candidates if n in ds.coords), None)
    lon = next((n for n in lon_candidates if n in ds.coords), None)
    if lat is None or lon is None:
        raise ValueError(
            f"Could not find lat/lon in coords: {list(ds.coords)}")
    return lat, lon


def stack_year_month_to_series(da_ym):
    da_stk = da_ym.stack(time=("year", "month")).sortby("time")
    idx = pd.to_datetime([
        f"{int(y):04d}-{int(m):02d}-01"
        for y, m in zip(da_stk["year"].values, da_stk["month"].values)
    ])
    return pd.Series(da_stk.values.astype(float), index=idx)


def area_weights_2d(ds, lat_name, lon_name):
    lat = ds[lat_name]
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    return w_lat.broadcast_like(ds[lat_name] * 0 + ds[lon_name] * 0 + 1)


def area_weighted_mean_monthly(da, w2d, lat_name, lon_name):
    num = (da * w2d).sum(dim=[lat_name, lon_name], skipna=True)
    den = (xr.where(np.isfinite(da), 1.0, np.nan) *
           w2d).sum(dim=[lat_name, lon_name], skipna=True)
    return stack_year_month_to_series(num / den)


def bootstrap_raf(values_event, values_normal, thr, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    e = np.asarray(values_event, dtype=float)
    n = np.asarray(values_normal, dtype=float)
    e = e[np.isfinite(e)]
    n = n[np.isfinite(n)]
    if len(e) < 8 or len(n) < 8:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    p_e = np.mean(e > thr)
    p_n = np.mean(n > thr)
    raf = p_e / max(p_n, EPS)

    boots = []
    for _ in range(n_boot):
        eb = rng.choice(e, size=len(e), replace=True)
        nb = rng.choice(n, size=len(n), replace=True)
        pe = np.mean(eb > thr)
        pn = np.mean(nb > thr)
        boots.append(pe / max(pn, EPS))
    boots = np.asarray(boots)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return raf, lo, hi, p_e, p_n


ds = xr.open_dataset(NC_PATH)
LAT, LON = guess_lat_lon_names(ds)
w2d = area_weights_2d(ds, LAT, LON)

sti = ds["sti_scdhi"]
sapei = ds["sapei_scdhi"]
scdhi = ds["scdhi"]

sti_z = rolling_mean(
    zscore(area_weighted_mean_monthly(sti, w2d, LAT, LON)), SMOOTH_MONTHS)
sapei_z = rolling_mean(
    zscore(area_weighted_mean_monthly(sapei, w2d, LAT, LON)), SMOOTH_MONTHS)
scdhi_z = rolling_mean(
    zscore(area_weighted_mean_monthly(scdhi, w2d, LAT, LON)), SMOOTH_MONTHS)

thr_sti = sti_z.quantile(P_HEAT)
thr_sapei = sapei_z.quantile(P_DROUGHT)
thr_scdhi = scdhi_z.quantile(P_COMP)
thr_norm = scdhi_z.quantile(P_NORMAL)

heat = sti_z >= thr_sti
drought = sapei_z <= thr_sapei
compound = scdhi_z <= thr_scdhi
heat_only = heat  # & (~drought)
drought_only = drought  # & (~heat)
normal = scdhi_z >= thr_norm

gdf = pd.read_csv(GEN_PATH)
if "TYPE OF PRODUCER" in gdf.columns:
    gdf = gdf[gdf["TYPE OF PRODUCER"] ==
              "Total Electric Power Industry"].copy()

gdf["Date"] = pd.to_datetime(gdf["YEAR"].astype(int).astype(
    str) + "-" + gdf["MONTH"].astype(int).astype(str) + "-01")
gen_wide = (
    gdf.pivot_table(index="Date", columns="ENERGY SOURCE",
                    values="GENERATION (Megawatthours)", aggfunc="sum")
    .fillna(0.0)
)
gen_wide["Fossil"] = gen_wide.get("Coal", 0) + gen_wide.get(
    "Natural Gas", 0) + gen_wide.get("Petroleum", 0) + gen_wide.get("Other Gases", 0)
gen_wide["Hydro"] = gen_wide.get("Hydroelectric Conventional", 0)
gen_wide["Solar"] = gen_wide.get("Solar Thermal and Photovoltaic", 0)
gen_wide["Wind"] = gen_wide.get("Wind", 0)
gen_wide["Nuclear"] = gen_wide.get("Nuclear", 0)
gen_wide["Other Renew"] = gen_wide.get("Geothermal", 0) + gen_wide.get(
    "Wood and Wood Derived Fuels", 0) + gen_wide.get("Other Biomass", 0)
gen_wide["Renewables"] = gen_wide["Hydro"] + \
    gen_wide["Solar"] + gen_wide["Wind"] + gen_wide["Other Renew"]
gen_wide["Total_MWh"] = gen_wide["Fossil"] + \
    gen_wide["Renewables"] + gen_wide["Nuclear"]
gen_wide.loc[gen_wide["Total_MWh"] <= 0, "Total_MWh"] = np.nan
gen_gwh = gen_wide["Total_MWh"] / 1000.0

ddf = pd.read_csv(DEM_PATH)
ddf["Date"] = pd.to_datetime(ddf["year"].astype(int).astype(
    str) + "-" + ddf["month"].astype(int).astype(str) + "-01")
if "total_sales" not in ddf.columns:
    raise ValueError(
        "Expected 'total_sales' in california_electricity_demand_monthly.csv")
dem_gwh = ddf.set_index("Date")["total_sales"].astype(float) / 1000.0

common = gen_gwh.index.intersection(dem_gwh.index).intersection(
    sti_z.index).intersection(sapei_z.index).intersection(scdhi_z.index)
gen_gwh = rolling_mean(gen_gwh.loc[common], SMOOTH_MONTHS)
dem_gwh = rolling_mean(dem_gwh.loc[common], SMOOTH_MONTHS)
sti_z = sti_z.loc[common]
sapei_z = sapei_z.loc[common]
scdhi_z = scdhi_z.loc[common]

imb_gwh = dem_gwh - gen_gwh
imb_z = zscore(imb_gwh)

heat_only = heat_only.reindex(common).fillna(False)
drought_only = drought_only.reindex(common).fillna(False)
compound = compound.reindex(common).fillna(False)
normal = normal.reindex(common).fillna(False)

events = {
    "Heatwave": heat_only,
    "Drought": drought_only,
    "Compound": compound
}
event_colors = {"Heatwave": C_HEAT, "Drought": C_DROUGHT, "Compound": C_COMP}

metrics_rows = []
for dt in common:
    metrics_rows.append({
        "Date": dt,
        "Demand_GWh": float(dem_gwh.loc[dt]),
        "Generation_GWh": float(gen_gwh.loc[dt]),
        "Imbalance_GWh": float(imb_gwh.loc[dt]),
        "Imbalance_z": float(imb_z.loc[dt]),
        "Heatwave_only": bool(heat_only.loc[dt]),
        "Drought_only": bool(drought_only.loc[dt]),
        "Compound": bool(compound.loc[dt]),
        "Normal": bool(normal.loc[dt]),
    })
pd.DataFrame(metrics_rows).to_csv(CSV_METRICS, index=False)

thr_risk = np.nanquantile(
    imb_gwh[normal.values], RISK_Q) if normal.sum() > 5 else np.nan

event_rows = []
boot_rows = []
for ev, mask in events.items():
    x_ev = imb_gwh[mask.values]
    x_nm = imb_gwh[normal.values]

    mean_ev = float(np.nanmean(x_ev))
    mean_nm = float(np.nanmean(x_nm))
    delta = mean_ev - mean_nm
    delta_pct = 100.0 * delta / mean_nm if mean_nm != 0 else np.nan

    tstat, pval = ttest_ind(x_ev.values, x_nm.values,
                            equal_var=False, nan_policy="omit")
    raf, lo, hi, p_e, p_n = bootstrap_raf(
        x_ev.values, x_nm.values, thr=thr_risk, n_boot=N_BOOT, seed=BOOT_SEED+hash(ev) % 1000)

    event_rows.append({
        "Event": ev,
        "N_event": int(mask.sum()),
        "N_normal": int(normal.sum()),
        "Mean_event_GWh": mean_ev,
        "Mean_normal_GWh": mean_nm,
        "Delta_GWh": float(delta),
        "Delta_pct_vs_normal": float(delta_pct),
        "tstat": float(tstat),
        "p_value": float(pval),
        "RiskThreshold_GWh_q90_normal": float(thr_risk),
        "P_imbalance_gt_thr_event": float(p_e) if np.isfinite(p_e) else np.nan,
        "P_imbalance_gt_thr_normal": float(p_n) if np.isfinite(p_n) else np.nan,
        "RAF": float(raf) if np.isfinite(raf) else np.nan,
        "RAF_CI95_lo": float(lo) if np.isfinite(lo) else np.nan,
        "RAF_CI95_hi": float(hi) if np.isfinite(hi) else np.nan,
    })

    if np.isfinite(raf):
        boot_rows.append({
            "Event": ev,
            "RAF": raf,
            "RAF_CI95_lo": lo,
            "RAF_CI95_hi": hi
        })

event_df = pd.DataFrame(event_rows)
boot_df = pd.DataFrame(boot_rows)
event_df.to_csv(CSV_EVENTS, index=False)
boot_df.to_csv(CSV_BOOT, index=False)

# =========================
# Helper styling (PNAS/Nature-ish)
# =========================


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def add_letter(ax, letter, x=0.01, y=0.98, fs=18):
    ax.text(x, y, letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=fs)


# =========================
# Figure layout: 2 rows x 3 cols
# =========================
fig = plt.figure(figsize=(16.5, 9.4))
gs = GridSpec(
    nrows=2, ncols=3, figure=fig,
    height_ratios=[1.25, 1.0],
    wspace=0.34, hspace=0.34
)
fig.subplots_adjust(left=0.055, right=0.985, top=0.97, bottom=0.10)

# -------------------------
# A) Signal (full-width)
# -------------------------
axA = fig.add_subplot(gs[0, :])

# Make the three curves visually separable:
# Demand/Generation thinner, Imbalance thicker (visual hierarchy)
axA.plot(common, zscore(dem_gwh), color=C_DEM,
         lw=2.1, alpha=0.85, label="Demand")
axA.plot(common, zscore(gen_gwh), color=C_GEN,
         lw=2.1, alpha=0.85, label="Generation")
axA.plot(common, imb_z,           color=C_IMB,
         lw=2.9, alpha=0.95, label="Imbalance")

axA.axhline(0, color="k", lw=1.0, alpha=0.35, zorder=0)

axA.set_ylabel("Standardized anomaly")
axA.set_xlabel("Time (Year)")

# ticks: major every 5y, minor every 1y (clean)
axA.xaxis.set_major_locator(mdates.YearLocator(5))
axA.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axA.xaxis.set_minor_locator(mdates.YearLocator(1))
axA.tick_params(axis="x", which="major", length=6, width=1.2)
axA.tick_params(axis="x", which="minor", length=3, width=0.9)
axA.tick_params(axis="y", which="major", length=6, width=1.2)

style_axis(axA)

# Legend: centered top-ish but not a title; clean
axA.legend(
    frameon=False, ncol=3,
    loc="lower left", bbox_to_anchor=(0.03, 0.05),
    # mode="expand",
    borderaxespad=0.0,
    # columnspacing=1.2, handlelength=2.6, handletextpad=0.6
)

add_letter(axA, "A")

# -------------------------
# B) Mean effect (Δ Imbalance)
# -------------------------
axB = fig.add_subplot(gs[1, 0])

order = ["Heatwave", "Drought", "Compound"]
vals = event_df.set_index(
    "Event").loc[order, "Delta_GWh"].to_numpy(dtype=float)
cols = [event_colors[e] for e in order]
x = np.arange(len(order))

axB.bar(
    x, vals,
    color=cols,
    alpha=0.90,
    width=0.70,
    edgecolor="black",
    linewidth=1.4
)
axB.axhline(0, color="k", lw=1.0, alpha=0.45)

axB.set_xticks(x)
axB.set_xticklabels(order, rotation=0)
axB.set_ylabel("Δ Imbalance vs normal (GWh)")

"""
# Optional: add subtle value labels (PNAS-ish, but not too loud)
for i, v in enumerate(vals):
    if np.isfinite(v):
        axB.text(i, v, f"{v:+.0f}", ha="center",
                 va="bottom" if v >= 0 else "top",
                 fontsize=12, color="#222222")
"""
style_axis(axB)
add_letter(axB, "B")

# -------------------------
# C) Distribution (Normal vs events)
# -------------------------
axC = fig.add_subplot(gs[1, 1])

x_all = imb_gwh[np.isfinite(imb_gwh.values)].values
xmin = np.nanpercentile(x_all, 2)
xmax = np.nanpercentile(x_all, 98)
grid = np.linspace(xmin, xmax, 260)

# Normal baseline
x_nm = imb_gwh[normal.values].dropna().values
if len(x_nm) > 10:
    axC.plot(grid, gaussian_kde(x_nm)(grid),
             color="#6f6f6f", lw=2.2, ls="--", label="Normal")

# Plot all three events to match story, but keep line weights consistent
for ev in order:
    m = events[ev].values
    x_ev = imb_gwh[m].dropna().values
    if len(x_ev) > 10:
        axC.plot(grid, gaussian_kde(x_ev)(grid),
                 color=event_colors[ev], lw=2.2, alpha=0.95, label=ev)

# Risk threshold line
if np.isfinite(thr_risk):
    axC.axvline(thr_risk, color="k", lw=1.0, alpha=0.55)

axC.set_xlabel("Imbalance (GWh)")
axC.set_ylabel("Probability density")

# Legend: 1 column, compact, not covering curves
axC.legend(frameon=False, ncol=1, loc="upper left",
           bbox_to_anchor=(0.02, 0.98))

style_axis(axC)
add_letter(axC, "C")

# -------------------------
# D) RAF (tail risk)
# -------------------------
axD = fig.add_subplot(gs[1, 2])

tmp = event_df.set_index("Event").loc[order]
raf = tmp["RAF"].to_numpy(dtype=float)
lo = tmp["RAF_CI95_lo"].to_numpy(dtype=float)
hi = tmp["RAF_CI95_hi"].to_numpy(dtype=float)

# horizontal with CIs
y = np.arange(len(order))
xerr = np.vstack([raf - lo, hi - raf])

axD.errorbar(
    raf, y, xerr=xerr,
    fmt="o", ms=7.5, lw=1.8, capsize=4,
    color="#222222", zorder=2
)
for i, ev in enumerate(order):
    axD.scatter(raf[i], i, s=95, color=event_colors[ev], zorder=3)

axD.axvline(1.0, color="k", lw=1.0, ls="--", alpha=0.6, zorder=1)

axD.set_yticks(y)
axD.set_yticklabels(order)
axD.set_xlabel("RAF")
axD.set_ylabel("")
axD.set_ylim(-0.6, len(order) - 0.4)

# tighten x-limits so there isn't huge whitespace
# (robust to CI)
finite = np.isfinite(np.r_[lo, hi, raf])
if finite.any():
    lo_min = float(np.nanmin(lo[finite[:len(lo)]])) if np.isfinite(
        lo).any() else np.nanmin(raf)
    hi_max = float(np.nanmax(hi[finite[len(lo):len(lo)+len(hi)]])
                   ) if np.isfinite(hi).any() else np.nanmax(raf)
    # fallback safer:
    lo_min = float(np.nanmin(lo)) if np.isfinite(
        lo).any() else float(np.nanmin(raf))
    hi_max = float(np.nanmax(hi)) if np.isfinite(
        hi).any() else float(np.nanmax(raf))
    pad = 0.12 * (hi_max - lo_min + 1e-9)
    axD.set_xlim(max(0.0, lo_min - pad), hi_max + pad)

style_axis(axD)
add_letter(axD, "D")

# -------------------------
# Save
# -------------------------
fig.savefig(FIG_PATH, bbox_inches="tight")
plt.close(fig)
# =========================
# Log
# =========================
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("=== Figure 6: Supply–Demand Imbalance (Core Figure) ===\n\n")
    f.write(
        f"Inputs:\n- NC: {NC_PATH}\n- GEN: {GEN_PATH}\n- DEM: {DEM_PATH}\n\n")
    f.write(
        f"Period: {common.min().date()} to {common.max().date()} | N={len(common)} months\n")
    f.write(f"Smoothing: {SMOOTH_MONTHS}-month centered rolling mean\n")
    f.write("Imbalance definition: Demand_GWh - Generation_GWh\n\n")
    f.write("Event thresholds (statewide, z-space):\n")
    f.write(f"- Heatwave: STI >= q{P_HEAT:.2f} (thr={thr_sti:.4f})\n")
    f.write(f"- Drought: SAPEI <= q{P_DROUGHT:.2f} (thr={thr_sapei:.4f})\n")
    f.write(f"- Compound: SCDHI <= q{P_COMP:.2f} (thr={thr_scdhi:.4f})\n")
    f.write(f"- Normal: SCDHI >= q{P_NORMAL:.2f} (thr={thr_norm:.4f})\n\n")
    f.write(
        f"Risk threshold: q{RISK_Q:.2f} of NORMAL imbalance = {thr_risk:.3f} GWh\n")
    f.write(f"Bootstrap RAF: N_BOOT={N_BOOT}, seed={BOOT_SEED}\n\n")
    f.write("Event metrics:\n")
    f.write(event_df.to_string(index=False))
    f.write("\n\nOutputs:\n")
    f.write(f"- Figure: {FIG_PATH}\n")
    f.write(f"- Metrics CSV: {CSV_METRICS}\n")
    f.write(f"- Event CSV: {CSV_EVENTS}\n")
    f.write(f"- Bootstrap CSV: {CSV_BOOT}\n")
