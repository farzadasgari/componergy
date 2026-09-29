"""
Systemic vulnerability figure: gridded climate stress composite (A),
regional demand vulnerability index (B), a generation-stress x
demand-surge risk surface (C), and its extreme-value return period (D).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

from componergy.figures.trend_stats import compute_area_weights, compute_statewide_mean
from componergy.figures.composite_stats import (
    series_to_data_array, compute_composite_table, benjamini_hochberg_fdr, zscore, deseasonalize_monthly,
    apply_fdr_to_grid, vectorized_welch_ttest_grid,
)
from componergy.figures.stats_log import write_figure_log
from componergy.analysis.events import classify_heatwave, classify_drought, classify_compound
from componergy.analysis.regions import build_county_to_region_map
from componergy.analysis.extremes import fit_gev_annual_maxima, fit_pot_gpd, return_period_of_value
from componergy.analysis.event_validation import validate_against_known_events, summarize_validation
from componergy.analysis.superadditivity import superadditivity_table
from componergy.analysis.placebo import placebo_table

SMOOTH_MONTHS = 3
ALPHA_SIG = 0.05

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "axes.linewidth": 1.4,
    "savefig.dpi": 600,
})

import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
