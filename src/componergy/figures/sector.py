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


def aggregate_consumption_by_sector(consumption_long_df: pd.DataFrame) -> pd.DataFrame:
    return (
            consumption_long_df.groupby(["Date", "Sector"])["Consumption (MWh)"]
            .sum()
            .unstack("Sector")
            .sort_index()
            / 1000.0
    )


def prepare_sector_data(indices_ds: xr.Dataset, consumption_long_df: pd.DataFrame,
                        sapei_var="sapei_3m", scdhi_var="scdhi_3m", smooth_months=3):
    """Classify statewide climate events and prepare statewide sector consumption in raw (GWh) and z-scored form."""
    sector_raw = aggregate_consumption_by_sector(consumption_long_df)
    sectors_order = [s for s in SECTOR_ORDER if s in sector_raw.columns] + \
                    [s for s in sector_raw.columns if s not in SECTOR_ORDER]
    sector_raw = sector_raw[sectors_order]

    weights = compute_area_weights(indices_ds["lat"])
    sti = zscore(compute_statewide_mean(indices_ds["sti"], weights).to_series())
    sapei = zscore(compute_statewide_mean(indices_ds[sapei_var], weights).to_series())
    scdhi = zscore(compute_statewide_mean(indices_ds[scdhi_var], weights).to_series())

    start = max(sector_raw.index.min(), sti.index.min(), sapei.index.min(), scdhi.index.min())
    end = min(sector_raw.index.max(), sti.index.max(), sapei.index.max(), scdhi.index.max())

    sector_raw = sector_raw.loc[start:end]
    sti, sapei, scdhi = sti.loc[start:end], sapei.loc[start:end], scdhi.loc[start:end]

    sector_sm = sector_raw.rolling(smooth_months, center=True, min_periods=1).mean()
    sector_local = sector_sm.apply(deseasonalize_monthly, axis=0)
    sector_z = sector_local.apply(zscore, axis=0)

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
        "sectors_order": sectors_order,
        "sector_raw": sector_sm,
        "sector_z": sector_z,
        "event_masks": event_masks,
        "normal": normal,
        "period_start": start,
        "period_end": end,
    }
