from componergy.paths import CALIFORNIA_MONTHLY_FILE, FIGURES_DIR
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NC_PATH = str(CALIFORNIA_MONTHLY_FILE)
OUT_DIR = str(FIGURES_DIR / "Figure 1")
os.makedirs(OUT_DIR, exist_ok=True)

FIG_PATH = os.path.join(OUT_DIR, "figure 1 - signals trends.jpg")
LOG_PATH = os.path.join(OUT_DIR, "figure 1 - signals trends.log")


SMOOTH_MONTHS = 3
ALPHA_SIG = 0.05

PC = ccrs.PlateCarree()
# CA_EXTENT = [-124.8, -113.95, 32.1, 42.45] # [-125, -113, 32, 43.7]


def california_extent(pad_deg=0.35, resolution="50m"):
    shp = shpreader.natural_earth(
        resolution=resolution,
        category="cultural",
        name="admin_1_states_provinces_lakes"
    )
    reader = shpreader.Reader(shp)
    ca_geom = None
    for rec in reader.records():
        if rec.attributes.get("admin") == "United States of America" and rec.attributes.get("name") == "California":
            ca_geom = rec.geometry
            break
    if ca_geom is None:
        raise RuntimeError(
            "Could not find California in Natural Earth admin_1 dataset.")
    minx, miny, maxx, maxy = ca_geom.bounds
    return [minx - pad_deg, maxx + pad_deg, miny - pad_deg, maxy + pad_deg]


CA_EXTENT = california_extent(pad_deg=0.35)


BASE_FONT = 18

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "dejavuserif",

    "font.size": BASE_FONT,
    "axes.labelsize": BASE_FONT,
    "xtick.labelsize": BASE_FONT - 1,
    "ytick.labelsize": BASE_FONT - 1,
    "legend.fontsize": BASE_FONT - 1,

    "axes.linewidth": 1.4,
    "savefig.dpi": 600,
})

C_STI = "crimson"
C_SAPEI = "deepskyblue"
C_SCDHI = "purple"
C_FOOT = "#8a8a8a"


def guess_lat_lon_names(dataset):
    lat_candidates = ["lat", "latitude", "y", "YLAT", "LAT"]
    lon_candidates = ["lon", "longitude", "x", "XLONG", "LON"]
    lat_name = next((n for n in lat_candidates if n in dataset.coords), None)
    lon_name = next((n for n in lon_candidates if n in dataset.coords), None)
    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Could not find lat/lon coords. Found coords: {list(dataset.coords)}")
    return lat_name, lon_name


def year_month_to_datetime_index(years, months):
    return pd.to_datetime([f"{int(y):04d}-{int(m):02d}-01" for y, m in zip(years, months)])


def stack_year_month_to_series(da_ym):
    da_stk = da_ym.stack(time=("year", "month")).sortby("time")
    idx = year_month_to_datetime_index(
        da_stk["year"].values, da_stk["month"].values)
    return pd.Series(da_stk.values.astype(float), index=idx)


def rolling_mean(s, w):
    return s.rolling(w, center=True, min_periods=1).mean()


def zscore(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if (sd and np.isfinite(sd)) else s * 0.0


def style_timeseries(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def style_map(ax):
    ax.set_extent(CA_EXTENT, crs=PC)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.8)
    ax.add_feature(cfeature.LAKES, linewidth=0.3, alpha=0.35)
    ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_letter(ax, letter, x=0.01, y=0.98, fs=26):

    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=fs, fontweight="normal",
        clip_on=False
    )


def area_weights_2d(ds, lat_name, lon_name):
    lat = ds[lat_name]
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w2d = w_lat.broadcast_like(ds[lat_name] * 0 + ds[lon_name] * 0 + 1)
    return w2d


def area_weighted_mean_monthly(da, w2d, lat_name, lon_name):
    num = (da * w2d).sum(dim=[lat_name, lon_name], skipna=True)
    den = (xr.where(np.isfinite(da), 1.0, np.nan) *
           w2d).sum(dim=[lat_name, lon_name], skipna=True)
    out = num / den
    return stack_year_month_to_series(out)


def area_fraction_extreme(da, q, tail, lat_name, lon_name):
    da_t = da.stack(time=("year", "month"))
    da_t = da_t.where(np.isfinite(da_t))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        thresh = da_t.quantile(q, dim="time", skipna=True)

    if tail == "upper":
        extreme = da_t >= thresh
    else:
        extreme = da_t <= thresh

    frac = extreme.mean(dim=[lat_name, lon_name], skipna=True)
    frac = frac.unstack("time")
    return stack_year_month_to_series(frac)


def annualize(da):
    return da.mean("month")


def trend_per_decade(da_annual):
    years = da_annual["year"].astype(float)
    x = years - years.mean("year")
    n = da_annual.sizes["year"]
    x2sum = (x ** 2).sum("year")

    slope_per_year = (da_annual * x).sum("year") / x2sum
    slope_per_decade = slope_per_year * 10.0

    pval = None
    if HAVE_SCIPY and n >= 3:
        ybar = da_annual.mean("year")
        yhat = ybar + slope_per_year * x
        resid = da_annual - yhat
        sse = (resid ** 2).sum("year")
        df = n - 2
        sigma2 = sse / df
        se_slope = np.sqrt(sigma2 / x2sum)
        tstat = slope_per_year / se_slope

        pval = xr.apply_ufunc(
            lambda t: 2.0 * (1.0 - stats.t.cdf(np.abs(t), df=df)),
            tstat,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

    return slope_per_decade, pval


def sym_limits_robust(da, q=0.98, fallback=1.0):
    vals = np.abs(da.values.ravel())
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return -fallback, fallback
    vmax = float(np.quantile(vals, q))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = fallback
    return -vmax, vmax


def summarize_series(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return {}
    return {
        "start": str(s.index.min().date()),
        "end": str(s.index.max().date()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "p10": float(np.percentile(s.values, 10)),
        "p90": float(np.percentile(s.values, 90)),
        "min": float(np.min(s.values)),
        "max": float(np.max(s.values)),
        "n_months": int(len(s))
    }


def summarize_trend_map(slope, p=None, alpha=0.05):
    vals = slope.values.ravel()
    vals = vals[np.isfinite(vals)]
    out = {
        "mean_slope_decade": float(np.mean(vals)) if len(vals) else np.nan,
        "median_slope_decade": float(np.median(vals)) if len(vals) else np.nan,
        "p10_slope_decade": float(np.percentile(vals, 10)) if len(vals) else np.nan,
        "p90_slope_decade": float(np.percentile(vals, 90)) if len(vals) else np.nan,
    }
    if p is not None:
        pv = p.values.ravel()
        m = np.isfinite(pv)
        out["sig_frac_p_lt_alpha"] = float(
            np.mean(pv[m] < alpha)) if m.sum() else np.nan
    return out


def slope_per_decade_1d(series):
    s = pd.to_numeric(series, errors="coerce").astype(float)
    m = np.isfinite(s.values)
    if m.sum() < 60:
        return np.nan
    years = series.index.year.values.astype(
        float)[m] + (series.index.month.values.astype(float)[m] - 0.5) / 12.0
    y = s.values[m]
    x = years - years.mean()
    slope_per_year = np.sum(y * x) / np.sum(x ** 2)
    return float(slope_per_year * 10.0)


def add_linear_trend_with_ci(
    ax, series,
    lw=1.8,
    alpha_band=0.2,
    line_color="0.35",
    band_color="0.35",
    z_line=8,
    z_band=7
):
    s = pd.to_numeric(series, errors="coerce").astype(float)
    m = np.isfinite(s.values)
    if m.sum() < 3:
        return

    idx = pd.DatetimeIndex(series.index[m])
    y = s.values[m].astype(float)

    years = idx.year.to_numpy(dtype=float) + \
        (idx.month.to_numpy(dtype=float) - 0.5) / 12.0
    x = years

    n = len(x)
    xbar = float(np.mean(x))
    ybar = float(np.mean(y))
    Sxx = float(np.sum((x - xbar) ** 2))
    if Sxx == 0.0:
        return

    slope = float(np.sum((x - xbar) * (y - ybar)) / Sxx)
    intercept = ybar - slope * xbar
    yhat = intercept + slope * x

    resid = y - yhat
    df = n - 2
    if df <= 0:
        tcrit = 1.96
        se_yhat = np.zeros_like(yhat)
    else:
        sse = float(np.sum(resid ** 2))
        sigma2 = sse / df
        se_yhat = np.sqrt(sigma2 * (1.0 / n + ((x - xbar) ** 2) / Sxx))
        tcrit = float(stats.t.ppf(0.975, df=df)) if HAVE_SCIPY else 1.96

    ci = tcrit * se_yhat

    ax.fill_between(idx, yhat - ci, yhat + ci, color=band_color,
                    alpha=alpha_band, linewidth=0, zorder=z_band)
    ax.plot(
        idx,
        yhat,
        color=line_color,
        lw=lw,
        linestyle="--",
        zorder=z_line
    )


ds = xr.open_dataset(NC_PATH)
LAT, LON = guess_lat_lon_names(ds)

sti = ds["sti_scdhi"]
sapei = ds["sapei_scdhi"]
scdhi = ds["scdhi"]

w2d = area_weights_2d(ds, LAT, LON)


sti_mean = area_weighted_mean_monthly(sti, w2d, LAT, LON)
sapei_mean = area_weighted_mean_monthly(sapei, w2d, LAT, LON)
scdhi_mean = area_weighted_mean_monthly(scdhi, w2d, LAT, LON)

sti_z = rolling_mean(zscore(sti_mean), SMOOTH_MONTHS)
sapei_z = rolling_mean(zscore(sapei_mean), SMOOTH_MONTHS)
scdhi_z = rolling_mean(zscore(scdhi_mean), SMOOTH_MONTHS)


sti_foot = rolling_mean(area_fraction_extreme(
    sti, q=0.90, tail="upper", lat_name=LAT, lon_name=LON), SMOOTH_MONTHS)
sapei_foot = rolling_mean(area_fraction_extreme(
    sapei, q=0.10, tail="lower", lat_name=LAT, lon_name=LON), SMOOTH_MONTHS)
scdhi_foot = rolling_mean(area_fraction_extreme(
    scdhi, q=0.10, tail="lower", lat_name=LAT, lon_name=LON), SMOOTH_MONTHS)


sti_ann = annualize(sti)
sapei_ann = annualize(sapei)
scdhi_ann = annualize(scdhi)

sti_slope, sti_p = trend_per_decade(sti_ann)
sapei_slope, sapei_p = trend_per_decade(sapei_ann)
scdhi_slope, scdhi_p = trend_per_decade(scdhi_ann)


vmin_sti, vmax_sti = sym_limits_robust(sti_slope)
vmin_sapei, vmax_sapei = sym_limits_robust(sapei_slope)
vmin_scdhi, vmax_scdhi = sym_limits_robust(scdhi_slope)


fig = plt.figure(figsize=(16.5, 8.6))
gs = fig.add_gridspec(
    nrows=3, ncols=3,
    width_ratios=[1.5, 1.5, 0.7],
    height_ratios=[1.0, 1.0, 1.0],
    wspace=0.03,
    hspace=0.22
)
fig.subplots_adjust(left=0.055, right=0.975, top=0.97, bottom=0.085)


def plot_signal_panel(ax, z_series, foot_series, color, ylabel, letter, foot_label):

    ax.plot(z_series.index, z_series.values, color=color, lw=2.2, zorder=4)
    ax.axhline(0, color="k", lw=1.8, alpha=0.35, zorder=1)
    ax.set_ylabel(ylabel)
    style_timeseries(ax)
    add_letter(ax, letter, x=0.01, y=0.98, fs=18)

    add_linear_trend_with_ci(
        ax,
        z_series,
        lw=3.0,
        alpha_band=0.18,
        line_color="black",
        band_color="black"
    )
    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_alpha(0)

    ax2.fill_between(
        foot_series.index,
        0.0,
        np.clip(foot_series.values, 0, 1),
        color=C_FOOT,
        alpha=0.22,
        linewidth=0,
        zorder=0
    )
    ax2.plot(
        foot_series.index,
        np.clip(foot_series.values, 0, 1),
        color=C_FOOT,
        lw=1.0,
        alpha=0.65,
        zorder=1
    )
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])

    for sp in ax2.spines.values():
        sp.set_visible(False)

    #     handles = [
    #     Line2D([0], [0], color=color, lw=2.2, label=ylabel),
    #     Line2D([0], [0], color="0.35", lw=1.8, label="Linear trend"),
    #     Patch(facecolor=C_FOOT, edgecolor="none", alpha=0.22, label=foot_label),
    # ]

    # ax.legend(
    #     handles=handles,
    #     frameon=False,
    #     ncol=3,
    #     loc="upper left",
    #     bbox_to_anchor=(0.08, 1.08),
    #     borderaxespad=0.0,
    #     columnspacing=1.2,
    #     handletextpad=0.6,
    #     handlelength=2.0,
    # )


axA = fig.add_subplot(gs[0, 0:2])
plot_signal_panel(
    axA, sti_z, sti_foot, C_STI,
    ylabel="STI",
    letter="A",
    foot_label="Heatwave footprint"
)


axB = fig.add_subplot(gs[1, 0:2], sharex=axA)
plot_signal_panel(
    axB, sapei_z, sapei_foot, C_SAPEI,
    ylabel="SAPEI",
    letter="B",
    foot_label="Drought footprint"
)


axC = fig.add_subplot(gs[2, 0:2], sharex=axA)
plot_signal_panel(
    axC, scdhi_z, scdhi_foot, C_SCDHI,
    ylabel="SCDHI",
    letter="C",
    foot_label="Compound footprint"
)
axC.set_xlabel("Time (Year)")

shared_handles = [
    Line2D([0], [0], color=C_STI,   lw=3.2, label="STI"),
    Line2D([0], [0], color=C_SAPEI, lw=3.2, label="SAPEI"),
    Line2D([0], [0], color=C_SCDHI, lw=3.2, label="SCDHI"),
    Line2D([0], [0], color="black", lw=3.0,
           linestyle="--", label="Linear trend"),
    Patch(
        facecolor=C_FOOT,
        edgecolor="none",
        alpha=0.22,
        label="Extreme-event footprint"
    ),
]

axA.legend(
    handles=shared_handles,
    frameon=False,
    ncol=len(shared_handles),
    loc="lower left",
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
    mode="expand",
    borderaxespad=0.0,
    columnspacing=0.8,
    handletextpad=0.4,
    handlelength=1.8,
)

for ax in [axA, axB, axC]:
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(5))
    ax.tick_params(axis="x", which="minor", length=0)

axA.tick_params(axis="x", labelbottom=False)
axB.tick_params(axis="x", labelbottom=False)


def plot_trend_map(ax, slope, pvals, vmin, vmax, cbar_label, letter):

    im = slope.plot(
        ax=ax, transform=PC, cmap="RdBu_r",
        vmin=vmin, vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={
            "label": cbar_label,
            "shrink": 0.88,
            "fraction": 0.1,
            "pad": 0.03
        }
    )
    style_map(ax)

    add_letter(ax, letter, x=-0.12, y=1.02, fs=18)

    if pvals is not None:
        sig = (pvals < ALPHA_SIG)
        step = 3
        sig_sub = sig.isel({LAT: slice(None, None, step),
                           LON: slice(None, None, step)})

        lat_vals = sig_sub[LAT].values
        lon_vals = sig_sub[LON].values
        LON2, LAT2 = np.meshgrid(lon_vals, lat_vals)
        mask = np.asarray(sig_sub.values, dtype=bool)

        ax.scatter(
            LON2[mask], LAT2[mask],
            s=2, c="k", alpha=0.25,
            transform=PC, linewidths=0
        )
    return im


axD = fig.add_subplot(gs[0, 2], projection=PC)
plot_trend_map(
    axD, sti_slope, sti_p,
    vmin=vmin_sti, vmax=vmax_sti,
    cbar_label="STI trend per decade",
    letter="D"
)

axE = fig.add_subplot(gs[1, 2], projection=PC)
plot_trend_map(
    axE, sapei_slope, sapei_p,
    vmin=vmin_sapei, vmax=vmax_sapei,
    cbar_label="SAPEI trend per decade",
    letter="E"
)

axF = fig.add_subplot(gs[2, 2], projection=PC)
plot_trend_map(
    axF, scdhi_slope, scdhi_p,
    vmin=vmin_scdhi, vmax=vmax_scdhi,
    cbar_label="SCDHI trend per decade",
    letter="F"
)

plt.savefig(FIG_PATH, bbox_inches="tight", dpi=600)
plt.close(fig)


y0 = int(ds["year"].min())
y1 = int(ds["year"].max())

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("=== Figure 1: Statewide signals + trends (CA) ===\n\n")
    f.write(f"NetCDF: {NC_PATH}\n")
    f.write(f"Year range: {y0}–{y1}\n")
    f.write(f"Smoothing: {SMOOTH_MONTHS}-month centered rolling mean\n")
    f.write("Signals: area-weighted mean with cos(latitude) weights.\n")
    f.write(
        "Footprints: fraction of grid cells meeting cellwise percentile thresholds.\n")
    f.write(
        f"Trend maps: annual mean trend per decade; significance shown as stippling at p<{ALPHA_SIG}.\n\n")

    f.write("---- Statewide signals (z-scored, smoothed) summary ----\n")
    f.write(f"STI_z:   {summarize_series(sti_z)}\n")
    f.write(f"SAPEI_z: {summarize_series(sapei_z)}\n")
    f.write(f"SCDHI_z: {summarize_series(scdhi_z)}\n\n")

    f.write("---- Statewide signal OLS trend (z-units per decade) ----\n")
    f.write(f"STI_z trend/decade:   {slope_per_decade_1d(sti_z):.6f}\n")
    f.write(f"SAPEI_z trend/decade: {slope_per_decade_1d(sapei_z):.6f}\n")
    f.write(f"SCDHI_z trend/decade: {slope_per_decade_1d(scdhi_z):.6f}\n\n")

    f.write("---- Footprint summaries (smoothed fraction 0–1) ----\n")
    f.write(f"Heatwave footprint (STI ≥ 90p): {summarize_series(sti_foot)}\n")
    f.write(
        f"Drought footprint  (SAPEI ≤ 10p): {summarize_series(sapei_foot)}\n")
    f.write(
        f"Compound footprint (SCDHI ≤ 10p): {summarize_series(scdhi_foot)}\n\n")

    f.write("---- Trend map summaries (per decade; full field) ----\n")
    f.write(
        f"STI slope stats:   {summarize_trend_map(sti_slope, sti_p, ALPHA_SIG)}\n")
    f.write(
        f"SAPEI slope stats: {summarize_trend_map(sapei_slope, sapei_p, ALPHA_SIG)}\n")
    f.write(
        f"SCDHI slope stats: {summarize_trend_map(scdhi_slope, scdhi_p, ALPHA_SIG)}\n")
