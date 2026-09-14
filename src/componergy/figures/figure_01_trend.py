"""Six-panel figure: statewide standardized signals with trend lines, and spatial trend maps."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
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
