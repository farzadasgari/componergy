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


def prepare_imbalance_data(indices_ds: xr.Dataset, generation_df: pd.DataFrame, demand_df: pd.DataFrame,
                           sapei_var="sapei_3m", scdhi_var="scdhi_3m", smooth_months=SMOOTH_MONTHS):
    """Align statewide demand, generation, and climate signals; classify events."""
    gen_gwh = generation_df["Total"] / 1000.0
    dem_gwh = demand_df["total_sales"] / 1000.0

    weights = compute_area_weights(indices_ds["lat"])
    sti = zscore(compute_statewide_mean(indices_ds["sti"], weights).to_series())
    sapei = zscore(compute_statewide_mean(indices_ds[sapei_var], weights).to_series())
    scdhi = zscore(compute_statewide_mean(indices_ds[scdhi_var], weights).to_series())
    sti = sti.rolling(smooth_months, center=True, min_periods=1).mean()
    sapei = sapei.rolling(smooth_months, center=True, min_periods=1).mean()
    scdhi = scdhi.rolling(smooth_months, center=True, min_periods=1).mean()

    common = gen_gwh.index.intersection(dem_gwh.index).intersection(sti.index).intersection(sapei.index).intersection(
        scdhi.index)

    gen_gwh = gen_gwh.loc[common].rolling(smooth_months, center=True, min_periods=1).mean()
    dem_gwh = dem_gwh.loc[common].rolling(smooth_months, center=True, min_periods=1).mean()
    sti, sapei, scdhi = sti.loc[common], sapei.loc[common], scdhi.loc[common]

    imb_gwh = dem_gwh - gen_gwh
    imb_z = zscore(imb_gwh)

    heat = classify_heatwave(series_to_data_array(sti)).to_series()
    drought = classify_drought(series_to_data_array(sapei)).to_series()
    compound = classify_compound(series_to_data_array(scdhi)).to_series()
    normal = ~(heat | drought | compound)

    event_masks = {
        "Heatwave": heat & (~drought),
        "Drought": drought & (~heat),
        "Compound": compound,
    }

    return {
        "common": common,
        "gen_gwh": gen_gwh, "dem_gwh": dem_gwh, "imb_gwh": imb_gwh, "imb_z": imb_z,
        "event_masks": event_masks, "normal": normal,
    }
