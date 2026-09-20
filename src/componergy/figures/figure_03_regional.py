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
