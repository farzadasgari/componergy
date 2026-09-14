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
