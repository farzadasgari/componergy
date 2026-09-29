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


def compute_gridded_vulnerability(data: dict, sapei_var="sapei_3m", scdhi_var="scdhi_3m") -> tuple:
    """Per-grid-cell composite climate stress: mean of STI-heat, -SAPEI-drought, -SCDHI-compound event anomalies."""
    indices_ds = data["indices_ds"]
    heat_mask, drought_mask = data["event_masks"]["Heatwave"], data["event_masks"]["Drought"]
    compound_mask, normal_mask = data["event_masks"]["Compound"], data["normal"]

    common_times = indices_ds["time"].to_index().intersection(heat_mask.index)

    sti_grid = indices_ds["sti"].sel(time=common_times)
    sapei_grid = indices_ds[sapei_var].sel(time=common_times)
    scdhi_grid = indices_ds[scdhi_var].sel(time=common_times)

    def as_time_da(mask):
        return xr.DataArray(mask.reindex(common_times).fillna(False).values, dims="time", coords={"time": common_times})

    heat_da, drought_da = as_time_da(heat_mask), as_time_da(drought_mask)
    compound_da, normal_da = as_time_da(compound_mask), as_time_da(normal_mask)

    def mean_over_mask(da, mask_da):
        if int(mask_da.sum()) == 0:
            return da.mean(dim="time") * np.nan
        return da.where(mask_da).mean(dim="time")

    delta_heat = mean_over_mask(sti_grid, heat_da) - mean_over_mask(sti_grid, normal_da)
    delta_drought = -(mean_over_mask(sapei_grid, drought_da) - mean_over_mask(sapei_grid, normal_da))
    delta_compound = -(mean_over_mask(scdhi_grid, compound_da) - mean_over_mask(scdhi_grid, normal_da))

    gridded = (delta_heat + delta_drought + delta_compound) / 3

    p_heat = vectorized_welch_ttest_grid(sti_grid, heat_da, normal_da)
    p_drought = vectorized_welch_ttest_grid(sapei_grid, drought_da, normal_da)
    p_compound = vectorized_welch_ttest_grid(scdhi_grid, compound_da, normal_da)

    q_heat = apply_fdr_to_grid(p_heat)
    q_drought = apply_fdr_to_grid(p_drought)
    q_compound = apply_fdr_to_grid(p_compound)

    is_significant = (q_heat < ALPHA_SIG) | (q_drought < ALPHA_SIG) | (q_compound < ALPHA_SIG)

    return gridded, is_significant


def compute_regional_evi(data: dict) -> tuple:
    """Per-region Event Vulnerability Index: mean Delta_z of demand across the three events, FDR-corrected supporting table."""
    comp = compute_composite_table(data["electricity_z"], data["event_masks"], data["normal"])
    comp = comp.rename(columns={"Source": "Region", "Difference": "Delta_z"})
    comp["q_value"] = benjamini_hochberg_fdr(comp["p_value"].values)

    evi = comp.pivot(index="Region", columns="Event", values="Delta_z").mean(axis=1).rename("EVI")
    return evi, comp


def compute_risk_surface(data: dict) -> pd.DataFrame:
    """System Risk = (-SCDHI) x (statewide demand z-score): a defined illustrative index, not a validated grid-reliability metric."""
    gen_stress = -data["scdhi"]
    demand_surge = data["demand_state_z"]
    common = gen_stress.index.intersection(demand_surge.index)
    gen_stress, demand_surge = gen_stress.loc[common], demand_surge.loc[common]

    monthly_risk = gen_stress * demand_surge
    return pd.DataFrame({"Gen_Stress": gen_stress, "Demand_Surge": demand_surge, "System_Risk": monthly_risk})


def compute_return_period(monthly_risk: pd.Series) -> pd.DataFrame:
    """Empirical return period of System Risk via the Weibull plotting position, (n+1)/rank, in years."""
    risk_series = monthly_risk.dropna().sort_values(ascending=False)
    n = len(risk_series)
    ranks = np.arange(1, n + 1)
    return_period_years = ((n + 1.0) / ranks) / 12.0

    return pd.DataFrame({
        "Date": risk_series.index,
        "System_Risk": risk_series.values,
        "Return_Period_Years": return_period_years,
    })


def region_abbrev(name: str) -> str:
    parts = [p for p in name.replace("-", " ").split() if p]
    return "".join(p[0].upper() for p in parts[:3])


def text_color_for_bg(value, norm, cmap_obj, light_thresh=0.55):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "black"
    r, g, b, _ = cmap_obj(norm(value))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if lum >= light_thresh else "white"


def plot_gridded_vulnerability_panel(ax, gridded: xr.DataArray, is_significant: xr.DataArray, county_boundary_dissolved,
                                     regions_gdf, cmap_obj, norm):
    """Composite climate stress map with FDR-corrected significance stippling."""
    lons = gridded["lon"].values
    lats = gridded["lat"].values
    XX, YY = np.meshgrid(lons, lats)

    ax.pcolormesh(XX, YY, gridded.values, cmap=cmap_obj, norm=norm, shading="auto", rasterized=True)
    county_boundary_dissolved.plot(ax=ax, color="black", linewidth=1.2)
    regions_gdf.plot(ax=ax, facecolor="none", edgecolor="#333333", linewidth=0.75, alpha=0.45)

    step = 3
    sig_sub = is_significant.isel(lat=slice(None, None, step), lon=slice(None, None, step))
    lon2, lat2 = np.meshgrid(sig_sub["lon"].values, sig_sub["lat"].values)
    mask = np.asarray(sig_sub.values, dtype=bool)
    ax.scatter(lon2[mask], lat2[mask], s=2, c="k", alpha=0.25, linewidths=0)

    ax.set_axis_off()


def plot_regional_evi_panel(ax, evi: pd.Series, regions_gdf, county_boundary_dissolved, cmap_obj, norm):
    regions_plot = regions_gdf.copy()
    regions_plot["abbr"] = regions_plot["Region"].map(region_abbrev)
    regions_plot["centroid"] = regions_plot.geometry.representative_point()
    regions_plot["EVI"] = regions_plot["Region"].map(evi.to_dict())

    regions_plot.plot(ax=ax, column="EVI", cmap=cmap_obj, norm=norm, linewidth=1.0, edgecolor="white")
    county_boundary_dissolved.plot(ax=ax, color="black", linewidth=1.2)

    for _, r in regions_plot.iterrows():
        if r["centroid"] is None or r["centroid"].is_empty:
            continue
        tcol = text_color_for_bg(r["EVI"], norm, cmap_obj, light_thresh=0.55)
        stroke_col = "black" if tcol == "white" else "white"
        ax.text(r["centroid"].x, r["centroid"].y, r["abbr"], ha="center", va="center",
                fontsize=16, fontweight="bold", color=tcol, alpha=0.98,
                path_effects=[pe.withStroke(linewidth=1.6, foreground=stroke_col), pe.Normal()])

    ax.set_axis_off()
