"""
Supply-demand imbalance figure: statewide Demand - Generation over time,
event-composite mean shifts, distributional comparison, and a bootstrap
Risk Attributable Fraction (RAF) forest plot -- how much more likely the
imbalance is to exceed a normal-period risk threshold during each event
type.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

from componergy.figures.trend_stats import compute_area_weights, compute_statewide_mean
from componergy.figures.composite_stats import (
    series_to_data_array, compute_composite_table, benjamini_hochberg_fdr, zscore,
)
from componergy.figures.stats_log import write_figure_log
from componergy.analysis.events import classify_heatwave, classify_drought, classify_compound
from componergy.analysis.block_bootstrap import block_bootstrap_raf
from componergy.analysis.superadditivity import superadditivity_test
from componergy.analysis.placebo import placebo_table
from componergy.analysis.system_boundary import validate_system_boundary, compute_import_fraction

SMOOTH_MONTHS = 3
RISK_Q = 0.90
N_BOOT = 5000
BOOT_SEED = 42
EPS = 1e-9

C_HEAT = "crimson"
C_DROUGHT = "deepskyblue"
C_COMP = "purple"
C_IMB = "#222222"
C_GEN = "#d95f02"
C_DEM = "#1f78b4"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "axes.linewidth": 1.4,
    "savefig.dpi": 600,
})

import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
