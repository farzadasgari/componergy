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
    series_to_data_array, compute_composite_table, benjamini_hochberg_fdr,
)
from componergy.figures.figure_03_regional import zscore
from componergy.figures.stats_log import write_figure_log
from componergy.analysis.events import classify_heatwave, classify_drought, classify_compound

Q_ALPHA_FDR = 0.05
Q_ALPHA_FDR_MARGINAL = 0.10

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


def prepare_zonal_data(indices_ds: xr.Dataset, consumption_long_df: pd.DataFrame, regions_gdf,
                       county_region_map: pd.DataFrame,
                       sapei_var="sapei_3m", scdhi_var="scdhi_3m", smooth_months=3):
    from componergy.figures.figure_03_regional import aggregate_consumption_to_regional

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
    sti, sapei, scdhi = sti.loc[start:end], sapei.loc[start:end], scdhi.loc[start:end]

    electricity_sm = electricity_raw.rolling(smooth_months, center=True, min_periods=1).mean()
    electricity_local = electricity_sm.apply(deseasonalize_monthly, axis=0)
    electricity_z = electricity_local.apply(zscore, axis=0)

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
        "electricity_raw": electricity_sm,
        "electricity_z": electricity_z,
        "event_masks": event_masks,
        "normal": normal,
        "period_start": start,
        "period_end": end,
    }


def compute_zonal_composites(data: dict) -> pd.DataFrame:
    z_table = compute_composite_table(data["electricity_z"], data["event_masks"], data["normal"])
    raw_table = compute_composite_table(data["electricity_raw"], data["event_masks"], data["normal"])

    merged = z_table.merge(
        raw_table[["Event", "Source", "Mean_event", "Mean_normal"]],
        on=["Event", "Source"], suffixes=("_z", "_raw"),
    )
    merged["Delta_pct"] = 100.0 * (merged["Mean_event_raw"] - merged["Mean_normal_raw"]) / merged["Mean_normal_raw"]
    merged["q_value"] = benjamini_hochberg_fdr(merged["p_value"].values)

    return merged.rename(columns={"Source": "Region", "Difference": "Delta_z"})


def region_abbrev(name: str) -> str:
    parts = [p for p in name.replace("-", " ").split() if p]
    return "".join(p[0].upper() for p in parts[:3])


def text_color_for_bg(value, norm, cmap_obj, light_thresh=0.55):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "black"
    r, g, b, _ = cmap_obj(norm(value))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if lum >= light_thresh else "white"
