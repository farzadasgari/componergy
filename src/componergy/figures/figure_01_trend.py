"""Six-panel figure: statewide standardized signals with trend lines, and spatial trend maps."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from componergy.download.boundaries import get_california_boundary
from componergy.figures.trend_stats import (
    compute_area_weights, compute_statewide_mean, compute_footprint_fraction,
    annualize_monthly, compute_trend_per_decade, compute_linear_trend_with_ci,
)
from componergy.figures.stats_log import write_figure_log
from componergy.analysis.events import classify_heatwave, classify_drought, classify_compound
from componergy.paths import INDICES_FILE, FIGURES_DIR

import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

PC = ccrs.PlateCarree()
SMOOTH_MONTHS = 3
ALPHA_SIG = 0.05

C_STI = "crimson"
C_SAPEI = "deepskyblue"
C_SCDHI = "purple"
C_FOOT = "#8a8a8a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 17,
    "axes.linewidth": 1.4,
    "savefig.dpi": 600,
})


def compute_extent_from_boundary(boundary, pad_deg=0.35):
    """Map extent [minx, maxx, miny, maxy] from a boundary GeoDataFrame's bounds, with padding in degrees."""
    minx, miny, maxx, maxy = boundary.total_bounds
    return [minx - pad_deg, maxx + pad_deg, miny - pad_deg, maxy + pad_deg]


def zscore(series):
    """Standardize a pandas Series to zero mean, unit variance. Returns all-zero if std is 0 or non-finite."""
    std = series.std(ddof=0)
    if not std or not np.isfinite(std):
        return series * 0.0
    return (series - series.mean()) / std


def style_timeseries(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def style_map(ax, extent, boundary):
    """
    Apply shared map styling: extent, coastline/borders/lakes/rivers, and the boundary outline.

    Note: cfeature.COASTLINE/BORDERS/LAKES/RIVERS fetch Natural Earth
    shapefiles over the network on first use (cached locally afterward).
    """
    ax.set_extent(extent, crs=PC)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, linewidth=0.3, alpha=0.35)
    ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.35)
    ax.add_geometries(boundary.geometry, crs=PC, facecolor="none", edgecolor="black", linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_letter(ax, letter, x=0.01, y=0.98, fs=18):
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top", fontsize=fs, clip_on=False)


def sym_limits_robust(da, q=0.98, fallback=1.0):
    """Symmetric (-vmax, vmax) color limits from the q-th percentile of absolute values, robust to outliers."""
    vals = np.abs(da.values.ravel())
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return -fallback, fallback
    vmax = float(np.quantile(vals, q))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = fallback
    return -vmax, vmax


def plot_signal_panel(ax, z_series, foot_series, fitted, ci_band, color, ylabel, letter):
    ax.plot(z_series.index, z_series.values, color=color, lw=2.2, zorder=4)
    ax.axhline(0, color="k", lw=1.8, alpha=0.35, zorder=1)
    ax.set_ylabel(ylabel)
    style_timeseries(ax)
    add_letter(ax, letter)

    ax.fill_between(fitted.index, fitted.values - ci_band.values, fitted.values + ci_band.values,
                    color="black", alpha=0.18, linewidth=0, zorder=7)
    ax.plot(fitted.index, fitted.values, color="black", lw=3.0, linestyle="--", zorder=8)

    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_alpha(0)
    ax2.fill_between(foot_series.index, 0.0, np.clip(foot_series.values, 0, 1),
                     color=C_FOOT, alpha=0.22, linewidth=0, zorder=0)
    ax2.plot(foot_series.index, np.clip(foot_series.values, 0, 1), color=C_FOOT, lw=1.0, alpha=0.65, zorder=1)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)


def plot_trend_map(ax, slope, pvals, vmin, vmax, cbar_label, letter, extent, boundary):
    im = slope.plot(ax=ax, transform=PC, cmap="RdBu_r", vmin=vmin, vmax=vmax, add_colorbar=False)
    cbar = plt.colorbar(im, ax=ax, shrink=0.88, fraction=0.18, pad=0.06)
    cbar.set_label(cbar_label, labelpad=14)
    style_map(ax, extent, boundary)
    add_letter(ax, letter, x=-0.12, y=1.02)

    sig = pvals < ALPHA_SIG
    step = 3
    sig_sub = sig.isel(lat=slice(None, None, step), lon=slice(None, None, step))
    lon2, lat2 = np.meshgrid(sig_sub["lon"].values, sig_sub["lat"].values)
    mask = np.asarray(sig_sub.values, dtype=bool)
    ax.scatter(lon2[mask], lat2[mask], s=2, c="k", alpha=0.25, transform=PC, linewidths=0)


def build_figure_01(indices_ds: xr.Dataset, sapei_var: str = "sapei_3m", scdhi_var: str = "scdhi_3m"):
    """
    Build the full 6-panel figure and a dict of summary statistics.

    Parameters
    ----------
    indices_ds : xr.Dataset
        Must contain 'sti' and the requested sapei_var/scdhi_var, on
        (time, lat, lon).
    sapei_var, scdhi_var : str
        Which SAPEI/SCDHI timescale column to use.

    Returns
    -------
    (fig, log_sections) : tuple[matplotlib.figure.Figure, dict]
    """
    boundary = get_california_boundary()
    extent = compute_extent_from_boundary(boundary)

    weights = compute_area_weights(indices_ds["lat"])

    sti_mean = compute_statewide_mean(indices_ds["sti"], weights).to_series()
    sapei_mean = compute_statewide_mean(indices_ds[sapei_var], weights).to_series()
    scdhi_mean = compute_statewide_mean(indices_ds[scdhi_var], weights).to_series()

    sti_z = zscore(sti_mean).rolling(SMOOTH_MONTHS, center=True, min_periods=1).mean()
    sapei_z = zscore(sapei_mean).rolling(SMOOTH_MONTHS, center=True, min_periods=1).mean()
    scdhi_z = zscore(scdhi_mean).rolling(SMOOTH_MONTHS, center=True, min_periods=1).mean()

    heat_mask = classify_heatwave(indices_ds["sti"])
    drought_mask = classify_drought(indices_ds[sapei_var])
    compound_mask = classify_compound(indices_ds[scdhi_var])

    sti_foot = compute_footprint_fraction(heat_mask).to_series().rolling(SMOOTH_MONTHS, center=True,
                                                                         min_periods=1).mean()
    sapei_foot = compute_footprint_fraction(drought_mask).to_series().rolling(SMOOTH_MONTHS, center=True,
                                                                              min_periods=1).mean()
    scdhi_foot = compute_footprint_fraction(compound_mask).to_series().rolling(SMOOTH_MONTHS, center=True,
                                                                               min_periods=1).mean()

    sti_fitted, sti_ci = compute_linear_trend_with_ci(sti_z)
    sapei_fitted, sapei_ci = compute_linear_trend_with_ci(sapei_z)
    scdhi_fitted, scdhi_ci = compute_linear_trend_with_ci(scdhi_z)

    sti_annual = annualize_monthly(indices_ds["sti"])
    sapei_annual = annualize_monthly(indices_ds[sapei_var])
    scdhi_annual = annualize_monthly(indices_ds[scdhi_var])

    sti_slope, sti_p = compute_trend_per_decade(sti_annual)
    sapei_slope, sapei_p = compute_trend_per_decade(sapei_annual)
    scdhi_slope, scdhi_p = compute_trend_per_decade(scdhi_annual)

    fig = plt.figure(figsize=(16.5, 8.6))
    gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[1.5, 1.5, 0.7],
                          height_ratios=[1.0, 1.0, 1.0], wspace=0.03, hspace=0.22)
    fig.subplots_adjust(left=0.055, right=0.975, top=0.93, bottom=0.085)

    axA = fig.add_subplot(gs[0, 0:2])
    plot_signal_panel(axA, sti_z, sti_foot, sti_fitted, sti_ci, C_STI, "STI", "A")

    axB = fig.add_subplot(gs[1, 0:2], sharex=axA)
    plot_signal_panel(axB, sapei_z, sapei_foot, sapei_fitted, sapei_ci, C_SAPEI, "SAPEI", "B")

    axC = fig.add_subplot(gs[2, 0:2], sharex=axA)
    plot_signal_panel(axC, scdhi_z, scdhi_foot, scdhi_fitted, scdhi_ci, C_SCDHI, "SCDHI", "C")
    axC.set_xlabel("Time (Year)")

    shared_handles = [
        Line2D([0], [0], color=C_STI, lw=3.2, label="STI"),
        Line2D([0], [0], color=C_SAPEI, lw=3.2, label="SAPEI"),
        Line2D([0], [0], color=C_SCDHI, lw=3.2, label="SCDHI"),
        Line2D([0], [0], color="black", lw=3.0, linestyle="--", label="Linear trend"),
        Patch(facecolor=C_FOOT, edgecolor="none", alpha=0.22, label="Extreme-event footprint"),
    ]
    axA.legend(handles=shared_handles, frameon=False, ncol=len(shared_handles), loc="lower left",
               bbox_to_anchor=(0.0, 1.02, 1.0, 0.2), mode="expand", borderaxespad=0.0,
               columnspacing=0.8, handletextpad=0.4, handlelength=1.8)

    for ax in [axA, axB, axC]:
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.YearLocator(5))
        ax.tick_params(axis="x", which="minor", length=0)
    axA.tick_params(axis="x", labelbottom=False)
    axB.tick_params(axis="x", labelbottom=False)

    vmin_sti, vmax_sti = sym_limits_robust(sti_slope)
    vmin_sapei, vmax_sapei = sym_limits_robust(sapei_slope)
    vmin_scdhi, vmax_scdhi = sym_limits_robust(scdhi_slope)

    axD = fig.add_subplot(gs[0, 2], projection=PC)
    plot_trend_map(axD, sti_slope, sti_p, vmin_sti, vmax_sti, "Trend per decade", "D", extent, boundary)

    axE = fig.add_subplot(gs[1, 2], projection=PC)
    plot_trend_map(axE, sapei_slope, sapei_p, vmin_sapei, vmax_sapei, "Trend per decade", "E", extent, boundary)

    axF = fig.add_subplot(gs[2, 2], projection=PC)
    plot_trend_map(axF, scdhi_slope, scdhi_p, vmin_scdhi, vmax_scdhi, "Trend per decade", "F", extent, boundary)

    log_sections = {
        "Overview": {
            "sapei_variable": sapei_var,
            "scdhi_variable": scdhi_var,
            "smoothing_months": SMOOTH_MONTHS,
            "significance_alpha": ALPHA_SIG,
            "year_start": int(indices_ds["time"].dt.year.min()),
            "year_end": int(indices_ds["time"].dt.year.max()),
        },
        "STI statewide signal": {"mean": float(sti_z.mean()), "std": float(sti_z.std()),
                                 "n_months": int(sti_z.notna().sum())},
        "SAPEI statewide signal": {"mean": float(sapei_z.mean()), "std": float(sapei_z.std()),
                                   "n_months": int(sapei_z.notna().sum())},
        "SCDHI statewide signal": {"mean": float(scdhi_z.mean()), "std": float(scdhi_z.std()),
                                   "n_months": int(scdhi_z.notna().sum())},
        "Heatwave footprint (STI >= 90th pct)": {"mean_fraction": float(sti_foot.mean()),
                                                 "max_fraction": float(sti_foot.max())},
        "Drought footprint (SAPEI <= 10th pct)": {"mean_fraction": float(sapei_foot.mean()),
                                                  "max_fraction": float(sapei_foot.max())},
        "Compound footprint (SCDHI <= 10th pct)": {"mean_fraction": float(scdhi_foot.mean()),
                                                   "max_fraction": float(scdhi_foot.max())},
        "STI trend map": {
            "mean_slope_per_decade": float(np.nanmean(sti_slope.values)),
            "fraction_significant": float(np.nanmean((sti_p < ALPHA_SIG).values)),
        },
        "SAPEI trend map": {
            "mean_slope_per_decade": float(np.nanmean(sapei_slope.values)),
            "fraction_significant": float(np.nanmean((sapei_p < ALPHA_SIG).values)),
        },
        "SCDHI trend map": {
            "mean_slope_per_decade": float(np.nanmean(scdhi_slope.values)),
            "fraction_significant": float(np.nanmean((scdhi_p < ALPHA_SIG).values)),
        },
    }

    return fig, log_sections


def main(sapei_var: str = "sapei_3m", scdhi_var: str = "scdhi_3m") -> None:
    """Load INDICES_FILE, build the figure, and write the .jpg, .log, and .json outputs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = FIGURES_DIR / "figure_01"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {INDICES_FILE}...")
    indices_ds = xr.open_dataset(INDICES_FILE)

    print("building figure 1...")
    fig, log_sections = build_figure_01(indices_ds, sapei_var=sapei_var, scdhi_var=scdhi_var)

    fig_path = out_dir / "figure_01_trend.jpg"
    fig.savefig(fig_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"wrote {fig_path}")

    write_figure_log(out_dir / "figure_01_trend", log_sections)
    print(f"wrote {out_dir / 'figure_01_trend.log'} and .json")


if __name__ == "__main__":
    main()
