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


def prepare_vulnerability_data(indices_ds: xr.Dataset, consumption_long_df: pd.DataFrame, regions_gdf,
                               county_region_map: pd.DataFrame,
                               sapei_var="sapei_3m", scdhi_var="scdhi_3m", smooth_months=SMOOTH_MONTHS):
    """Align regional demand, statewide demand, and statewide climate signals; classify events."""
    from componergy.figures.regional import aggregate_consumption_to_regional

    electricity_raw = aggregate_consumption_to_regional(consumption_long_df, county_region_map)
    regions_order = [r for r in regions_gdf["Region"] if r in electricity_raw.columns]
    electricity_raw = electricity_raw[regions_order]

    weights = compute_area_weights(indices_ds["lat"])
    sti = zscore(compute_statewide_mean(indices_ds["sti"], weights).to_series())
    sapei = zscore(compute_statewide_mean(indices_ds[sapei_var], weights).to_series())
    scdhi = zscore(compute_statewide_mean(indices_ds[scdhi_var], weights).to_series())

    start = max(electricity_raw.index.min(), sti.index.min(), sapei.index.min(), scdhi.index.min())
    end = min(electricity_raw.index.max(), sti.index.max(), sapei.index.max(), scdhi.index.max())

    electricity_raw = electricity_raw.loc[start:end]
    sti = sti.loc[start:end].rolling(smooth_months, center=True, min_periods=1).mean()
    sapei = sapei.loc[start:end].rolling(smooth_months, center=True, min_periods=1).mean()
    scdhi = scdhi.loc[start:end].rolling(smooth_months, center=True, min_periods=1).mean()

    electricity_sm = electricity_raw.rolling(smooth_months, center=True, min_periods=1).mean()
    electricity_local = electricity_sm.apply(deseasonalize_monthly, axis=0)
    electricity_z = electricity_local.apply(zscore, axis=0)

    demand_state_gwh = electricity_sm.sum(axis=1)
    demand_state_z = zscore(deseasonalize_monthly(demand_state_gwh))

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
        "regions_order": regions_order,
        "electricity_z": electricity_z,
        "demand_state_z": demand_state_z,
        "sti": sti, "sapei": sapei, "scdhi": scdhi,
        "event_masks": event_masks, "normal": normal,
        "indices_ds": indices_ds,
        "period_start": start, "period_end": end,
    }
