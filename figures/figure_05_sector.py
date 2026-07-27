import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind
import os
import sys
from pathlib import Path
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from componergy.paths import COUNTY_CONSUMPTION_XLSX, CALIFORNIA_MONTHLY_FILE, FIGURES_DIR

consumption_path = str(COUNTY_CONSUMPTION_XLSX)
netcdf_path = str(CALIFORNIA_MONTHLY_FILE)

OUT_DIR = str(FIGURES_DIR / "Figure 5")
os.makedirs(OUT_DIR, exist_ok=True)

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
    idx = pd.to_datetime(
        [f"{int(yy):04d}-{int(mm):02d}-01" for yy, mm in zip(y, m)])
    return da_1d, idx


def statewide_driver(ds_window: xr.Dataset, varname: str) -> pd.Series:
    lat = ds_window["lat"]
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    da = ds_window[varname].weighted(w).mean(dim=["lat", "lon"])
    da_1d, idx = to_time_index_from_year_month(da)
    return pd.Series(da_1d.values, index=idx).sort_index()


def sector_abbrev(name: str) -> str:
    manual = {
        "Residential": "Residential",
        "Commercial": "Commercial",
        "Industrial": "Industrial",
        "Mining": "Mining",
        "Streetlighting": "SL",
        "Transportation, Communication, and Utilities": "TCU",
        "TCU": "TCU",
        "Ag With Water Pump": "AgPump",
        "Agriculture With Water Pump": "AgPump",
    }
    for k, v in manual.items():
        if k.lower() == str(name).lower():
            return v
    return str(name)[:7]


df = pd.read_excel(consumption_path, sheet_name="Electricity Consumption")
df["Date"] = pd.to_datetime(df["Year"].astype(
    str) + "-" + df["Month"].astype(str) + "-01")

if "Sector" not in df.columns:
    raise ValueError(
        "Expected a 'Sector' column in the Excel sheet (long format).")
if "Consumption (MWh)" not in df.columns:
    raise ValueError(
        "Expected column 'Consumption (MWh)' not found in the Excel sheet.")

df["Sector"] = df["Sector"].astype("string").str.strip()

sector_state = (
    df.groupby(["Date", "Sector"])["Consumption (MWh)"]
      .sum()
      .unstack("Sector")
      .sort_index()
    / 1000.0   # -> GWh
)

preferred = ["Residential", "Commercial", "Industrial", "Ag With Water Pump",
             "Transportation, Communication, and Utilities", "Mining", "Streetlighting"]
sectors = [s for s in preferred if s in sector_state.columns] + \
    [s for s in sector_state.columns if s not in preferred]
sector_state = sector_state[sectors]

ds = xr.open_dataset(netcdf_path)
ds_window = ds.sel(year=slice(2008, 2024))

sti_state = statewide_driver(ds_window, "sti_scdhi")
sapei_state = statewide_driver(ds_window, "sapei_scdhi")
scdhi_state = statewide_driver(ds_window, "scdhi")

start = max(sector_state.index.min(), sti_state.index.min(),
            sapei_state.index.min(), scdhi_state.index.min())
end = min(sector_state.index.max(), sti_state.index.max(),
          sapei_state.index.max(), scdhi_state.index.max())

sector_state = sector_state.loc[start:end]
sti_state = sti_state.loc[start:end]
sapei_state = sapei_state.loc[start:end]
scdhi_state = scdhi_state.loc[start:end]

smooth_window = 3
S = sector_state.rolling(smooth_window, center=True, min_periods=1).mean()
STI = sti_state.rolling(smooth_window, center=True, min_periods=1).mean()
SAPEI = sapei_state.rolling(smooth_window, center=True, min_periods=1).mean()
SCDHI = scdhi_state.rolling(smooth_window, center=True, min_periods=1).mean()

S_local = deseasonalize_monthly(S)
S_local_z = zscore_frame(S_local)

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
normal = SCDHI >= thr_norm

heat_only = heat & (~drought)
drought_only = drought # & (~heat)

event_defs = {
    "Heatwave": heat_only,
    "Drought": drought_only,
    "Compound": compound
}
events = ["Heatwave", "Drought", "Compound"]

N_normal = int(normal.sum())
N_events = {k: int(v.sum()) for k, v in event_defs.items()}

rows = []
norm_mask = normal.reindex(S.index).fillna(False)

for sec in S.columns:
    y_raw = S[sec].dropna()
    y_z = S_local_z[sec].reindex(y_raw.index).dropna()

    nm = norm_mask.reindex(y_z.index).fillna(False)
    yN_raw = y_raw.reindex(y_z.index)[nm.values]
    yN_z = y_z[nm.values]

    meanN_raw = float(np.nanmean(yN_raw.values))
    meanN_z = float(np.nanmean(yN_z.values))

    for ev, mask in event_defs.items():
        m = mask.reindex(y_z.index).fillna(False)
        yE_raw = y_raw.reindex(y_z.index)[m.values]
        yE_z = y_z[m.values]

        tstat, pval = ttest_ind(yE_z.values, yN_z.values,
                                equal_var=False, nan_policy="omit")

        meanE_raw = float(np.nanmean(yE_raw.values))
        meanE_z = float(np.nanmean(yE_z.values))

        delta_gwh = meanE_raw - meanN_raw
        delta_pct = 100.0 * delta_gwh / meanN_raw if meanN_raw != 0 else np.nan
        delta_z = meanE_z - meanN_z

        rows.append({
            "Sector": sec,
            "SectorAbbr": sector_abbrev(sec),
            "Event": ev,
            "N_event": int(m.sum()),
            "N_normal": int(nm.sum()),
            "Mean_event_GWh": meanE_raw,
            "Mean_normal_GWh": meanN_raw,
            "Delta_GWh": delta_gwh,
            "Delta_pct": delta_pct,
            "Delta_z": delta_z,
            "tstat": float(tstat),
            "p_value": float(pval),
        })

comp = pd.DataFrame(rows)
comp.to_csv(os.path.join(
    OUT_DIR, "figure 5 - sector composites.csv"), index=False)

with open(os.path.join(OUT_DIR, "figure 5 - sector composites.log"), "w", encoding="utf-8") as f:
    f.write("=== Figure 5: Sector fingerprint (statewide) ===\n\n")
    f.write(
        f"Period: {start.date()} to {end.date()}  | months={len(S.index)}\n")
    f.write(f"Smoothing: {smooth_window}-month centered rolling mean\n")
    f.write("Sector anomalies: deseasonalized monthly climatology removed, then z-scored per sector\n\n")
    f.write("Event thresholds (statewide):\n")
    f.write(f"  Heat: STI ≥ q{q_heat:.2f} (thr={thr_heat:.3f})\n")
    f.write(f"  Drought: SAPEI ≤ q{q_drought:.2f} (thr={thr_drought:.3f})\n")
    f.write(
        f"  Compound-index: SCDHI ≤ q{q_compound:.2f} (thr={thr_comp:.3f})\n")
    f.write(f"  Normal: SCDHI ≥ q{q_normal:.2f} (thr={thr_norm:.3f})\n\n")
    f.write(f"Counts: Normal={N_normal} | " +
            " | ".join([f"{k}={v}" for k, v in N_events.items()]) + "\n\n")

mat_g = comp.pivot(index="Sector", columns="Event",
                   values="Delta_GWh").reindex(index=sectors, columns=events)
mat_z = comp.pivot(index="Sector", columns="Event",
                   values="Delta_z").reindex(index=sectors, columns=events)
mat_p = comp.pivot(index="Sector", columns="Event",
                   values="p_value").reindex(index=sectors, columns=events)

abs_g = np.abs(mat_g.values)
gmax = float(np.nanpercentile(abs_g[np.isfinite(abs_g)], 95)) if np.isfinite(
    abs_g).any() else 1.0
gmax = max(1e-6, gmax)

vals_z = mat_z.values
vmax_z = float(np.nanpercentile(
    np.abs(vals_z[np.isfinite(vals_z)]), 90)) if np.isfinite(vals_z).any() else 1.0
vmax_z = max(0.5, vmax_z)

cmap = "RdBu_r"
normz = Normalize(vmin=-vmax_z, vmax=vmax_z)
cmap_obj = plt.get_cmap(cmap)

n_sec = len(sectors)
theta = np.linspace(0, 2*np.pi, n_sec, endpoint=False)
width = (2*np.pi / n_sec) * 0.88

ring_gap = 0.12
ring_thick_max = 0.55
ring_base = 0.80

ring_start = {
    "Heatwave": ring_base,
    "Drought":  ring_base + (ring_thick_max + ring_gap),
    "Compound": ring_base + 2*(ring_thick_max + ring_gap),
}

fig = plt.figure(figsize=(13.8, 9.0), constrained_layout=False)
# gs = GridSpec(nrows=1, ncols=3, width_ratios=[1.15, 0.3, 0.05], figure=fig)

# ax = fig.add_subplot(gs[0, 0], projection="polar")
# ax_leg = fig.add_subplot(gs[0, 1])
# ax_leg.axis("off")
# ax_cbar = fig.add_subplot(gs[0, 2])

gs = GridSpec(nrows=1, ncols=2, width_ratios=[1.22, 0.24], figure=fig)

ax = fig.add_subplot(gs[0, 0], projection="polar")
ax_leg = fig.add_subplot(gs[0, 1])
ax_leg.axis("off")

ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi/2)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.grid(alpha=0.12, linewidth=0.8)
ax.spines["polar"].set_visible(False)

for ev in events:
    base = ring_start[ev]
    for i, sec in enumerate(sectors):
        dg = mat_g.loc[sec, ev]
        dz = mat_z.loc[sec, ev]
        pv = mat_p.loc[sec, ev]
        if pd.isna(dg) or pd.isna(dz):
            continue

        thick = (min(abs(dg)/gmax, 1.0) ** 0.85) * ring_thick_max
        face = cmap_obj(normz(dz))

        if pd.notna(pv) and pv < 0.05:
            ec, lw = "black", 1.4
        elif pd.notna(pv) and pv < 0.10:
            ec, lw = "#444444", 1.1
        else:
            ec, lw = "none", 0.0

        ax.bar(theta[i], thick, width=width, bottom=base,
               color=face, edgecolor=ec, linewidth=lw, align="edge", alpha=0.97)


# outer = ring_start["Compound"] + ring_thick_max + 0.25
r_outer_ring = ring_start["Compound"] + ring_thick_max
outer = r_outer_ring + 0.06

ax.set_ylim(0, outer + 0.10)
for i, sec in enumerate(sectors):
    lab = sector_abbrev(sec)
    ang = theta[i] + width/2

    rot_deg = np.degrees(ang)
    if 90 < rot_deg < 270:
        ha = "right"
    else:
        ha = "left"

    ax.text(ang, outer, lab,
            fontsize=14, fontweight="bold",
            rotation=0, ha=ha, va="center",
            clip_on=False)


for ev in events:
    ax.text(np.deg2rad(2), ring_start[ev] + ring_thick_max*0.55,
            f"{ev}", fontsize=16, #fontweight="bold",
            ha="left", va="center", color="white",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")])


totals = comp.groupby("Event")["Delta_GWh"].sum().reindex(events)
center_txt = "Total ΔGWh vs normal\n" + \
    "\n".join([f"{ev}: {totals[ev]:+.1f}" for ev in events])

ax.text(0.5, 0.5, center_txt,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=14)


sm = ScalarMappable(norm=normz, cmap=cmap)
sm.set_array([])

# Put colorbar inside the legend panel (former significance area)
# [x0, y0, width, height] in ax_leg axes coordinates
cax = ax_leg.inset_axes([0.08, 0.08, 0.18, 0.46])

cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
cbar.set_label("Δz sector anomaly", labelpad=8, fontsize=16)
cbar.outline.set_linewidth(0.8)
cbar.ax.tick_params(width=0.8, length=3, labelsize=14)


ref = np.nanpercentile(abs_g[np.isfinite(abs_g)], [
                       50, 80, 95]) if np.isfinite(abs_g).any() else [1, 2, 3]
ref = np.unique(np.round(ref, 1))
y0 = 0.85
ax_leg.text(0.02, 0.92, "Contribution magnitude", #\n(wedge thickness ∝ |ΔGWh|)",
            transform=ax_leg.transAxes, fontsize=14, va="top") # , fontweight="bold")

ref = np.nanpercentile(abs_g[np.isfinite(abs_g)], [
                       50, 80, 95]) if np.isfinite(abs_g).any() else [1, 2, 3]
ref = np.unique(np.round(ref, 1))

# y = 0.78
# for r in ref[:3]:
#     thick = (min(r/gmax, 1.0) ** 0.85) * ring_thick_max
#     ax_leg.add_patch(plt.Rectangle((0.06, y-0.02), 0.25, 0.04, transform=ax_leg.transAxes,
#                                    facecolor="#dddddd", edgecolor="#333333", linewidth=1.0))
#     ax_leg.add_patch(plt.Rectangle((0.06, y-0.02), 0.25, 0.04*(thick/ring_thick_max), transform=ax_leg.transAxes,
#                                    facecolor="#777777", edgecolor="none"))
#     ax_leg.text(0.36, y, f"|ΔGWh| ≈ {r:.1f}",
#                 transform=ax_leg.transAxes, va="center", fontsize=12)
#     y -= 0.10

y = 0.82
for r in ref[:3]:
    thick = (min(r/gmax, 1.0) ** 0.85) * ring_thick_max

    ax_leg.add_patch(plt.Rectangle(
        (0.06, y-0.018), 0.25, 0.036,
        transform=ax_leg.transAxes,
        facecolor="#dddddd", edgecolor="#333333", linewidth=1.0
    ))
    ax_leg.add_patch(plt.Rectangle(
        (0.06, y-0.018), 0.25, 0.036*(thick/ring_thick_max),
        transform=ax_leg.transAxes,
        facecolor="#777777", edgecolor="none"
    ))
    ax_leg.text(
        0.34, y, f"|ΔGWh| ≈ {r:.1f}",
        transform=ax_leg.transAxes,
        va="center", fontsize=14
    )

    y -= 0.075 

# ax_leg.text(0.02, y-0.02, "Significance",
#             transform=ax_leg.transAxes, fontsize=12.5, fontweight="bold", va="top")

# ax_leg.add_patch(plt.Rectangle((0.06, y-0.12), 0.10, 0.05, transform=ax_leg.transAxes,
#                                facecolor="white", edgecolor="black", linewidth=1.4))
# ax_leg.text(0.20, y-0.095, "p < 0.05",
#             transform=ax_leg.transAxes, va="center", fontsize=12)

# ax_leg.add_patch(plt.Rectangle((0.06, y-0.20), 0.10, 0.05, transform=ax_leg.transAxes,
#                                facecolor="white", edgecolor="#444444", linewidth=1.1))
# ax_leg.text(0.20, y-0.175, "0.05 ≤ p < 0.10",
#             transform=ax_leg.transAxes, va="center", fontsize=12)

fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.05, wspace=0.02)
plt.savefig(os.path.join(
    OUT_DIR, "figure 5 - sector fingerprint.jpg"), bbox_inches="tight")
# plt.savefig(os.path.join(
#     OUT_DIR, "figure 5 - sector fingerprint.pdf"), bbox_inches="tight")
plt.close()

