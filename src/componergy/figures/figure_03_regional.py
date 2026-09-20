"""
Regional generation-climate-electricity figure: heatmaps of standardized
anomalies by climate region over time, and a slope-bar summary of each
region's dominant climate driver.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from componergy.analysis.regions import build_county_to_region_map, compute_regional_mean
from componergy.figures.print_style import SOURCE_STYLE
from componergy.figures.composite_stats import best_lag_slope, benjamini_hochberg_fdr, compute_incremental_r2
from componergy.figures.stats_log import write_figure_log

SMOOTH_MONTHS = 3
LAGS = range(-6, 7)
DRIVER_COLORS = {"STI": "crimson", "SAPEI": "deepskyblue", "SCDHI": "purple"}

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

import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not std or not np.isfinite(std):
        return series * 0.0
    return (series - series.mean()) / std


def aggregate_consumption_to_regional(consumption_long_df: pd.DataFrame,
                                      county_region_map: pd.DataFrame) -> pd.DataFrame:
    """Sum county consumption (all sectors, all planning-area splits) to monthly regional totals, in GWh."""
    merged = consumption_long_df.merge(county_region_map, on="CountyName", how="left")

    unmapped = sorted(merged.loc[merged["Region"].isna(), "CountyName"].unique().tolist())
    if unmapped:
        raise ValueError(f"Counties missing region assignment: {unmapped}")

    regional = (
            merged.groupby(["Date", "Region"])["Consumption (MWh)"]
            .sum()
            .unstack("Region")
            .sort_index()
            / 1000.0
    )
    return regional


def prepare_data(indices_ds: xr.Dataset, consumption_long_df: pd.DataFrame, regions_gdf,
                 county_region_map: pd.DataFrame,
                 sapei_var="sapei_3m", scdhi_var="scdhi_3m"):
    """
    Align regional electricity consumption with regional climate signals.

    Returns a dict with regions_order (list) and four smoothed,
    z-scored, common-period DataFrames (time index, one column per
    region): electricity_z, sti_z, sapei_z, scdhi_z, plus period_start/end.
    """
    electricity_regional = aggregate_consumption_to_regional(consumption_long_df, county_region_map)

    sti_regional = compute_regional_mean(indices_ds["sti"], regions_gdf)
    sapei_regional = compute_regional_mean(indices_ds[sapei_var], regions_gdf)
    scdhi_regional = compute_regional_mean(indices_ds[scdhi_var], regions_gdf)

    regions_order = [r for r in regions_gdf["Region"] if r in electricity_regional.columns]

    start = max(electricity_regional.index.min(), sti_regional.index.min(),
                sapei_regional.index.min(), scdhi_regional.index.min())
    end = min(electricity_regional.index.max(), sti_regional.index.max(),
              sapei_regional.index.max(), scdhi_regional.index.max())

    electricity_regional = electricity_regional.loc[start:end, regions_order]
    sti_regional = sti_regional.loc[start:end, regions_order]
    sapei_regional = sapei_regional.loc[start:end, regions_order]
    scdhi_regional = scdhi_regional.loc[start:end, regions_order]

    def smooth_and_z(df):
        smoothed = df.rolling(SMOOTH_MONTHS, center=True, min_periods=1).mean()
        return smoothed.apply(zscore, axis=0)

    return {
        "regions_order": regions_order,
        "electricity_z": smooth_and_z(electricity_regional),
        "sti_z": smooth_and_z(sti_regional),
        "sapei_z": smooth_and_z(sapei_regional),
        "scdhi_z": smooth_and_z(scdhi_regional),
        "period_start": start,
        "period_end": end,
    }


def plot_heatmap(ax, data, regions_order, row_label, vlim=2.5, cmap="viridis"):
    """
    One heatmap panel: regions on the y-axis, time on the x-axis.

    NaN cells (a region with no overlapping grid cells for this
    variable, see analysis.regions.compute_regional_mean) render as a
    distinct light gray rather than the colormap's own out-of-range
    color or the page's white background, so missing data reads as
    missing rather than as a value or blank space.
    """
    mat = data[regions_order].T.values
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#dddddd")
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=-vlim, vmax=vlim, cmap=cmap_obj)
    ax.set_yticks(np.arange(len(regions_order)))
    ax.set_yticklabels(regions_order)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(1.02, 0.5, row_label, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=16)
    return im


def build_heatmap_figure(data):
    regions_order = data["regions_order"]
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(13, 1.1 * len(regions_order) + 6), sharex=True)
    fig.subplots_adjust(left=0.16, right=0.83, top=0.96, bottom=0.06, hspace=0.15)

    im0 = plot_heatmap(axes[0], data["sti_z"], regions_order, "STI")
    plot_heatmap(axes[1], data["sapei_z"], regions_order, "SAPEI")
    plot_heatmap(axes[2], data["scdhi_z"], regions_order, "SCDHI")
    plot_heatmap(axes[3], data["electricity_z"], regions_order, "Electricity")

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Time (Year)")

    tick_idx = np.arange(0, len(data["electricity_z"].index), 24)
    tick_labels = [data["electricity_z"].index[i].strftime("%Y") for i in tick_idx]
    for ax in axes:
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels)

    cbar = fig.colorbar(im0, ax=axes, orientation="vertical", fraction=0.05, pad=0.03, shrink=0.9)
    cbar.set_label("Standardized anomaly (z-score)", labelpad=14)

    return fig


def compute_dominance_and_lag_summary(data):
    """
    Best-lag driver regression per (region, driver), with FDR-corrected q-values.

    One best_lag_slope() call per (region, driver) pair; the resulting
    p-values are corrected together across the whole table via
    benjamini_hochberg_fdr(), not read individually against 0.05.
    """
    regions_order = data["regions_order"]
    drivers = {"STI": data["sti_z"], "SAPEI": data["sapei_z"], "SCDHI": data["scdhi_z"]}

    rows = []
    for region in regions_order:
        y = data["electricity_z"][region].dropna()
        for driver_name, driver_df in drivers.items():
            x = driver_df[region].reindex(y.index)
            best = best_lag_slope(x, y, lags=LAGS, min_n=24)
            if best is None:
                continue
            rows.append({"Region": region, "Driver": driver_name, **best})

    summary = pd.DataFrame(rows)
    summary["q_value"] = benjamini_hochberg_fdr(summary["p_value"].values)
    return summary
