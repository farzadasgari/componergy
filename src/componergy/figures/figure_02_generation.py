"""Generation-mix-by-source figure: statewide timeseries and composite event-anomaly bars, one output file per source."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from componergy.figures.trend_stats import compute_area_weights, compute_statewide_mean
from componergy.figures.print_style import SOURCE_STYLE, EVENT_STYLE
from componergy.figures.composite_stats import series_to_data_array, compute_composite_table, best_lag_slope
from componergy.figures.stats_log import write_figure_log
from componergy.analysis.events import (
    classify_heatwave, classify_drought, classify_compound,
    normal_mask_exclude_all_events,
)
from componergy.paths import INDICES_FILE, GENERATION_MONTHLY_FILE, FIGURES_DIR

SMOOTH_MONTHS = 3
LAGS = range(-6, 7)
SOURCE_COLUMNS = ["Fossil", "Hydro", "Solar", "Wind", "Nuclear", "Other Renewable"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 17,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "axes.linewidth": 1.4,
    "savefig.dpi": 600,
})


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not std or not np.isfinite(std):
        return series * 0.0
    return (series - series.mean()) / std


def deseasonalize_monthly(series: pd.Series) -> pd.Series:
    clim = series.groupby(series.index.month).mean()
    return series - series.index.month.map(clim)


def style_timeseries_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(5))


def style_bar_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def add_letter(ax, letter, x=0.01, y=0.98, fs=19):
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top", fontsize=fs, clip_on=False)


def prepare_data(indices_ds: xr.Dataset, generation_df: pd.DataFrame, sapei_var="sapei_3m", scdhi_var="scdhi_3m"):
    weights = compute_area_weights(indices_ds["lat"])
    sti = zscore(compute_statewide_mean(indices_ds["sti"], weights).to_series())
    sapei = zscore(compute_statewide_mean(indices_ds[sapei_var], weights).to_series())
    scdhi = zscore(compute_statewide_mean(indices_ds[scdhi_var], weights).to_series())

    mix_pct = generation_df[SOURCE_COLUMNS].div(generation_df["Total"], axis=0) * 100.0
    mix_pct_sm = mix_pct.rolling(SMOOTH_MONTHS, center=True, min_periods=1).mean()

    common = mix_pct_sm.index.intersection(sti.index).intersection(sapei.index).intersection(scdhi.index)
    mix_pct_sm = mix_pct_sm.loc[common]
    sti, sapei, scdhi = sti.loc[common], sapei.loc[common], scdhi.loc[common]

    mix_z_total = mix_pct_sm.apply(zscore, axis=0)
    mix_z_local = mix_pct_sm.apply(deseasonalize_monthly, axis=0).apply(zscore, axis=0)

    heat = classify_heatwave(series_to_data_array(sti)).to_series()
    drought = classify_drought(series_to_data_array(sapei)).to_series()
    compound = classify_compound(series_to_data_array(scdhi)).to_series()

    heat_only = heat & (~drought)
    drought_only = drought & (~heat)
    normal = ~(heat | drought | compound)

    event_masks = {"Heatwave": heat_only, "Drought": drought_only, "Compound": compound}

    return {
        "drivers": {"STI": sti, "SAPEI": sapei, "SCDHI": scdhi},
        "mix_pct_sm": mix_pct_sm,
        "mix_z_total": mix_z_total,
        "mix_z_local": mix_z_local,
        "event_masks": event_masks,
        "normal": normal,
        "compound_months": mix_pct_sm.index[compound.reindex(mix_pct_sm.index).fillna(False).values],
    }


def shade_compound_months(ax, compound_months):
    for d in compound_months:
        ax.axvspan(d, d + pd.offsets.MonthEnd(0), color="#d73027", alpha=0.14, linewidth=0)


def plot_mix_timeseries(ax, data):
    for col in SOURCE_COLUMNS:
        style = SOURCE_STYLE[col]
        ax.plot(data["mix_pct_sm"].index, data["mix_pct_sm"][col].values,
                color=style["color"], linestyle=style["linestyle"], marker=style["marker"],
                markevery=24, markersize=6, lw=2.2, label=col)
    ax.set_ylabel("Share (%)")
    ax.set_xlabel("Time (Year)")
    shade_compound_months(ax, data["compound_months"])
    style_timeseries_axis(ax)

    ax.legend(ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.34),
              borderaxespad=0.0, handlelength=2.6, columnspacing=1.2)


def plot_single_source_timeseries(ax, data, src):
    style = SOURCE_STYLE[src]
    s = data["mix_pct_sm"][src]
    ax.plot(s.index, s.values, color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], markevery=24, markersize=6, lw=2.4)
    ax.set_ylabel(f"{src} share (%)")
    ax.set_xlabel("Time (Year)")
    shade_compound_months(ax, data["compound_months"])
    style_timeseries_axis(ax)
