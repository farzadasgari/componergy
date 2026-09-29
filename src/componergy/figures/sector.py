"""
Sector fingerprint figure: a polar wheel showing each consumption
sector's response to statewide climate events, ring thickness encoding
magnitude and color encoding direction, one ring per event type.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

from componergy.figures.trend_stats import compute_area_weights, compute_statewide_mean
from componergy.figures.composite_stats import (
    series_to_data_array, compute_composite_table, benjamini_hochberg_fdr, zscore, deseasonalize_monthly,
)
from componergy.figures.stats_log import write_figure_log
from componergy.analysis.events import classify_heatwave, classify_drought, classify_compound
from componergy.analysis.superadditivity import superadditivity_table
from componergy.analysis.placebo import placebo_table

Q_ALPHA_FDR = 0.05
Q_ALPHA_FDR_MARGINAL = 0.10

SECTOR_ABBREV = {
    "Residential": "Residential",
    "Commercial": "Commercial",
    "Industrial": "Industrial",
    "Ag With Water Pump": "AgPump",
    "TCU": "TCU",
    "Mining": "Mining",
    "Streetlighting": "SL",
}
SECTOR_ORDER = ["Residential", "Commercial", "Industrial", "Ag With Water Pump", "TCU", "Mining", "Streetlighting"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "axes.linewidth": 1.4,
    "savefig.dpi": 600,
})

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
