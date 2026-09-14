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