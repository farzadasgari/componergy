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
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

from componergy.figures.trend_stats import compute_area_weights, compute_statewide_mean
from componergy.figures.composite_stats import (
    benjamini_hochberg_fdr, zscore, deseasonalize_monthly,
    apply_fdr_to_grid, vectorized_welch_ttest_grid,
)
from componergy.figures.stats_log import write_figure_log
from componergy.figures.labels import region_abbrev, region_label_table, format_region_key
from componergy.analysis.events import build_event_masks
from componergy.analysis.regression import adjusted_composite_table
from componergy.figures.composite_outputs import write_composite_inference, INFERENCE_NOTE
from componergy.analysis.regions import build_county_to_region_map
from componergy.analysis.extremes import (
    fit_gev_annual_maxima, fit_pot_gpd, return_period_of_value,
    bootstrap_return_period, gev_goodness_of_fit, threshold_stability, stationarity_check,
)
from componergy.analysis.risk_sensitivity import (
    build_risk_variants, compare_risk_variants, double_calm_share,
)
from componergy.analysis.event_validation import validate_against_known_events, summarize_validation
from componergy.figures.print_style import SERIF_FONTS

SMOOTH_MONTHS = 3
ALPHA_SIG = 0.05

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": SERIF_FONTS,
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
                               sapei_var="sapei_3m", scdhi_var="scdhi_3m", smooth_months=SMOOTH_MONTHS,
                               baseline="full_record"):
    """Align regional demand, statewide demand, and statewide climate signals; classify events."""
    from componergy.figures.regional import aggregate_consumption_to_regional

    electricity_raw = aggregate_consumption_to_regional(consumption_long_df, county_region_map)
    regions_order = [r for r in regions_gdf["Region"] if r in electricity_raw.columns]
    electricity_raw = electricity_raw[regions_order]

    weights = compute_area_weights(indices_ds["lat"])
    sti = zscore(compute_statewide_mean(indices_ds["sti"], weights).to_series())
    sapei = zscore(compute_statewide_mean(indices_ds[sapei_var], weights).to_series())
    scdhi = zscore(compute_statewide_mean(indices_ds[scdhi_var], weights).to_series())
    sti_full, sapei_full, scdhi_full = sti, sapei, scdhi

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

    event_masks, normal, event_meta, event_flags = build_event_masks(
        sti_full, sapei_full, scdhi_full, window_index=sti.index,
        smooth_months=smooth_months, baseline=baseline, return_flags=True,
    )

    return {
        "regions_order": regions_order,
        "electricity_raw": electricity_sm,
        "electricity_z": electricity_z,
        "demand_state_z": demand_state_z,
        "scdhi_full": scdhi_full,
        "event_meta": event_meta, "event_flags": event_flags,
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


def compute_regional_evi(data: dict, trend: str = "linear") -> tuple:
    """Per-region Event Vulnerability Index: mean Delta_z of demand across the three events."""
    comp = adjusted_composite_table(data["electricity_raw"], data["event_masks"], data["normal"], trend=trend)
    comp = comp.rename(columns={"Source": "Region"})
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


def text_color_for_bg(value, norm, cmap_obj, light_thresh=0.55):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "black"
    r, g, b, _ = cmap_obj(norm(value))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if lum >= light_thresh else "white"


def add_letter(ax, letter, x=-0.08, y=0.96, fs=26):
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=fs, fontweight="normal", color="#222222")


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


def plot_risk_surface_panel(ax, risk_df: pd.DataFrame):
    gen_stress, demand_surge, monthly_risk = risk_df["Gen_Stress"], risk_df["Demand_Surge"], risk_df["System_Risk"]

    ax.scatter(gen_stress, demand_surge, c=monthly_risk, cmap="RdBu_r",
               norm=Normalize(vmin=monthly_risk.min(), vmax=monthly_risk.max()),
               s=70, alpha=0.88, edgecolors="none")

    x_range = np.linspace(gen_stress.min() - 0.2, gen_stress.max() + 0.2, 90)
    y_range = np.linspace(demand_surge.min() - 0.2, demand_surge.max() + 0.2, 90)
    Xg, Yg = np.meshgrid(x_range, y_range)
    Zg = Xg * Yg
    risk_levels = np.linspace(0.5, monthly_risk.max() * 0.85, 6)
    risk_levels = risk_levels[risk_levels > 0.2]
    if len(risk_levels) > 0:
        cs = ax.contour(Xg, Yg, Zg, levels=risk_levels, colors="#222222", linewidths=2, linestyles="dashed", alpha=0.85)
        ax.clabel(cs, inline=True, fontsize=14, fmt="Risk=%.1f", colors="#222222")

    ax.axvline(0, color="black", lw=1.1, alpha=0.45)
    ax.axhline(0, color="black", lw=1.1, alpha=0.45)
    ax.set_xlabel("Generation stress ($-$SCDHI)")
    ax.set_ylabel("Demand surge (z-score)")

    return monthly_risk


def plot_return_period_panel(ax, rp_df: pd.DataFrame):
    ax.plot(rp_df["Return_Period_Years"], rp_df["System_Risk"], "o-", color="#AF0A0A",
            markersize=8, linewidth=3.4, alpha=0.92)

    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("System risk index")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1", "2", "5", "10", "20"])

    max_rp = rp_df["Return_Period_Years"].iloc[0]
    max_risk = rp_df["System_Risk"].iloc[0]
    ax.annotate(f"Observed max\nRP = {max_rp:.1f} yr", xy=(max_rp, max_risk),
                xytext=(max_rp * 0.65, max_risk * 1.18), fontsize=17, color="#8B0000",
                arrowprops=dict(arrowstyle="->", color="#8B0000", lw=1.9))
    ax.grid(True, alpha=0.15)


def build_vulnerability_figure(gridded: xr.DataArray, is_significant: xr.DataArray, evi: pd.Series,
                               risk_df: pd.DataFrame, rp_df: pd.DataFrame,
                               regions_gdf, county_boundary_dissolved, comp: pd.DataFrame):
    delta_z_all = comp["Delta_z"].dropna().values
    grid_abs = np.abs(gridded.values.flatten())
    grid_abs = grid_abs[np.isfinite(grid_abs)]
    vmax_evi = max(0.5, float(np.nanpercentile(np.abs(delta_z_all), 90)),
                   float(np.nanpercentile(grid_abs, 90)) if len(grid_abs) > 0 else 0.5)

    cmap_obj = plt.get_cmap("RdBu_r")
    norm_map = Normalize(vmin=-vmax_evi, vmax=vmax_evi)

    fig = plt.figure(figsize=(16.2, 11.8))
    gs = GridSpec(nrows=2, ncols=2, figure=fig, width_ratios=[1.08, 1.0], height_ratios=[1.05, 1.0], wspace=0.55,
                  hspace=0.12)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.96, bottom=0.07)

    axA = fig.add_subplot(gs[0, 0])
    plot_gridded_vulnerability_panel(axA, gridded, is_significant, county_boundary_dissolved, regions_gdf, cmap_obj,
                                     norm_map)
    add_letter(axA, "A", x=-0.08)
    cax_A = fig.add_axes([0.455, 0.57, 0.013, 0.36])
    sm_A = ScalarMappable(norm=Normalize(vmin=float(gridded.min()), vmax=float(gridded.max())), cmap=cmap_obj)
    fig.colorbar(sm_A, cax=cax_A).set_label("Climate vulnerability")

    axB = fig.add_subplot(gs[0, 1])
    plot_regional_evi_panel(axB, evi, regions_gdf, county_boundary_dissolved, cmap_obj, norm_map)
    add_letter(axB, "B", x=-0.08)
    cax_B = fig.add_axes([0.905, 0.57, 0.013, 0.36])
    sm_B = ScalarMappable(norm=Normalize(vmin=float(evi.min()), vmax=float(evi.max())), cmap=cmap_obj)
    fig.colorbar(sm_B, cax=cax_B).set_label("Demand vulnerability")

    axC = fig.add_subplot(gs[1, 0])
    monthly_risk = plot_risk_surface_panel(axC, risk_df)
    add_letter(axC, "C", x=0.02)
    cax_risk = axC.inset_axes([1.07, 0.08, 0.028, 0.75])
    sm_risk = ScalarMappable(norm=Normalize(monthly_risk.min(), monthly_risk.max()), cmap="RdBu_r")
    fig.colorbar(sm_risk, cax=cax_risk).set_label("System risk", labelpad=8)

    axD = fig.add_subplot(gs[1, 1])
    plot_return_period_panel(axD, rp_df)
    add_letter(axD, "D", x=0.02)

    return fig


def load_generation_z():
    """Deseasonalized z-score of smoothed statewide in-state generation, or None if no generation file exists."""
    from componergy.paths import GENERATION_MONTHLY_FILE
    if not GENERATION_MONTHLY_FILE.exists():
        return None
    g = pd.read_csv(GENERATION_MONTHLY_FILE, parse_dates=[0], index_col=0)
    g.index = pd.DatetimeIndex(g.index.values, name=None)
    total = g["Total"].rolling(SMOOTH_MONTHS, center=True, min_periods=1).mean()
    return zscore(deseasonalize_monthly(total))


def compute_return_period_analysis(risk_series: pd.Series, hazard_full: pd.Series | None = None,
                                   n_boot: int = 500) -> dict:
    """Return-period analysis of the observed maximum, with the uncertainty and diagnostics it needs."""
    observed_max = float(risk_series.max())
    peak_date = risk_series.idxmax()
    gev = fit_gev_annual_maxima(risk_series)
    pot = fit_pot_gpd(risk_series)
    pot_dec = fit_pot_gpd(risk_series, decluster_run=3)
    rp_raw = return_period_of_value(observed_max, gev, pot)
    rp_dec = return_period_of_value(observed_max, gev, pot_dec)
    boot = bootstrap_return_period(risk_series, observed_max, n_boot=n_boot)

    level_rows = []
    for fit in (gev, pot, pot_dec):
        for T, lvl in sorted(fit.get("return_levels", {}).items()):
            level_rows.append({"Method": fit["method"], "Return_Period_Years": T, "Return_Level": lvl})

    n = int(risk_series.notna().sum())
    summary = {
        "peak_month": str(peak_date.date()), "observed_max_system_risk": observed_max,
        "n_months": n, "n_years": round(n / 12.0, 1),
        "empirical_return_period_years": (n + 1.0) / 12.0,
        "GEV_return_period_years": rp_raw["gev_return_period_years"],
        "POT_return_period_years": rp_raw["pot_return_period_years"],
        "POT_declustered_return_period_years": rp_dec["pot_return_period_years"],
        "bootstrap_CI95_GEV": [boot["gev"]["p2_5"], boot["gev"]["p97_5"]],
        "bootstrap_CI95_POT_declustered": [boot["pot_declustered"]["p2_5"], boot["pot_declustered"]["p97_5"]],
        "bootstrap_median_GEV": boot["gev"]["median"],
        "bootstrap_block_months": boot["block_months"], "bootstrap_n_boot": boot["n_boot"],
        "GEV_goodness_of_fit": gev_goodness_of_fit(risk_series),
        "stationarity": stationarity_check(risk_series),
        "caution": (
            "The EMPIRICAL return period of the record maximum equals (n+1)/12 years by construction, "
            "so it is fixed by the record length and says nothing about how extreme the maximum is. "
            "Quote a fitted value WITH its interval. The bootstrap interval is indicative, not exact "
            "(block bootstraps are not calibrated on very short records), and with this record length "
            "any return level beyond ~20-30 years is an extrapolation."
        ),
    }

    hazard = None
    if hazard_full is not None and peak_date in hazard_full.index:
        hv = float(hazard_full.loc[peak_date])
        h_gev = fit_gev_annual_maxima(hazard_full)
        h_pot = fit_pot_gpd(hazard_full, decluster_run=3)
        h_rp = return_period_of_value(hv, h_gev, h_pot)
        h_boot = bootstrap_return_period(hazard_full, hv, n_boot=max(100, n_boot // 2))
        hs = hazard_full.dropna()
        rank = int((hs > hv).sum()) + 1
        hazard = {
            "hazard_definition": "-SCDHI (statewide, area-weighted, 3-month smoothed); higher = more compound stress",
            "month": str(peak_date.date()), "hazard_value_z": hv,
            "record_months": int(len(hs)), "record_years": round(len(hs) / 12.0, 1),
            "rank_in_record": rank, "empirical_return_period_years": ((len(hs) + 1.0) / rank) / 12.0,
            "GEV_return_period_years": h_rp["gev_return_period_years"],
            "POT_declustered_return_period_years": h_rp["pot_return_period_years"],
            "bootstrap_CI95_GEV": [h_boot["gev"]["p2_5"], h_boot["gev"]["p97_5"]],
            "bootstrap_CI95_POT_declustered": [h_boot["pot_declustered"]["p2_5"], h_boot["pot_declustered"]["p97_5"]],
            "GEV_goodness_of_fit": gev_goodness_of_fit(hazard_full),
            "stationarity": stationarity_check(hazard_full),
        }
    return {"levels": pd.DataFrame(level_rows), "stability": threshold_stability(risk_series),
            "summary": summary, "hazard": hazard}


def main(sapei_var: str = "sapei_3m", scdhi_var: str = "scdhi_3m", baseline: str = "full_record") -> None:
    """Build the 4-panel vulnerability figure, its CSVs, and the log/json."""
    from componergy.download.climate_regions import get_california_climate_regions
    from componergy.download.boundaries import get_california_counties
    from componergy.preprocessing.consumption_cec import load_raw_consumption
    from componergy.netcdf_io import atomic_to_csv
    from componergy.paths import INDICES_FILE, COUNTY_CONSUMPTION_XLSX, FIGURES_DIR

    out_dir = FIGURES_DIR / "figure_07"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading climate regions, counties, and consumption data...")
    regions_gdf = get_california_climate_regions()
    counties_gdf = get_california_counties()
    county_region_map = build_county_to_region_map(counties_gdf, regions_gdf)
    consumption_long_df = load_raw_consumption(COUNTY_CONSUMPTION_XLSX)
    indices_ds = xr.open_dataset(INDICES_FILE)
    county_boundary_dissolved = counties_gdf.dissolve().boundary

    print("preparing vulnerability data...")
    data = prepare_vulnerability_data(indices_ds, consumption_long_df, regions_gdf, county_region_map,
                                      sapei_var=sapei_var, scdhi_var=scdhi_var, baseline=baseline)

    print("computing gridded vulnerability, EVI, risk surface, and return period...")
    gridded, is_significant = compute_gridded_vulnerability(data, sapei_var=sapei_var, scdhi_var=scdhi_var)
    evi, comp = compute_regional_evi(data)
    risk_df = compute_risk_surface(data)
    rp_df = compute_return_period(risk_df["System_Risk"])

    print("saving figure...")
    fig = build_vulnerability_figure(gridded, is_significant, evi, risk_df, rp_df, regions_gdf,
                                     county_boundary_dissolved, comp)
    fig.savefig(out_dir / "figure_07_systemic_vulnerability.jpg", bbox_inches="tight")
    plt.close(fig)

    atomic_to_csv(comp, out_dir / "figure_07_regional_evi_composites.csv", index=False)
    atomic_to_csv(risk_df, out_dir / "figure_07_monthly_system_risk.csv")
    atomic_to_csv(rp_df.head(10), out_dir / "figure_07_top_extremes.csv", index=False)

    print("computing amplification, robustness and placebo tables...")
    inference = write_composite_inference(data["electricity_raw"], data["event_masks"], data["normal"],
                                          out_dir, "figure_07", label="Region")
    label_table = region_label_table(data["regions_order"])
    atomic_to_csv(label_table, out_dir / "figure_07_region_abbreviations.csv", index=False)

    print("return-period analysis (fits, bootstrap uncertainty, diagnostics)...")
    risk_series = risk_df["System_Risk"]
    hazard_full = -(data["scdhi_full"].rolling(SMOOTH_MONTHS, center=True, min_periods=1).mean())
    rpa = compute_return_period_analysis(risk_series, hazard_full=hazard_full)
    atomic_to_csv(rpa["levels"], out_dir / "figure_07_return_levels.csv", index=False)
    atomic_to_csv(rpa["stability"], out_dir / "figure_07_return_period_threshold_stability.csv", index=False)
    if rpa["hazard"] is not None:
        atomic_to_csv(pd.DataFrame([{k: (str(v) if isinstance(v, (list, dict)) else v)
                                     for k, v in rpa["hazard"].items()}]),
                      out_dir / "figure_07_hazard_return_period.csv", index=False)

    print("risk-index sensitivity and external validation...")
    generation_z = load_generation_z()
    variants = build_risk_variants(-data["scdhi"], data["demand_state_z"], generation_z=generation_z)
    sensitivity = compare_risk_variants(variants)
    calm = double_calm_share(-data["scdhi"], data["demand_state_z"])
    atomic_to_csv(variants, out_dir / "figure_07_risk_variants.csv")
    atomic_to_csv(sensitivity, out_dir / "figure_07_risk_sensitivity.csv", index=False)

    validation_df = validate_against_known_events(risk_series)
    validation_summary = summarize_validation(validation_df)
    atomic_to_csv(validation_df, out_dir / "figure_07_known_event_validation.csv", index=False)
    by_variant = []
    for name in variants.columns:
        s = summarize_validation(validate_against_known_events(variants[name]))
        by_variant.append({"Variant": name, **s})
    validation_by_variant = pd.DataFrame(by_variant)
    atomic_to_csv(validation_by_variant, out_dir / "figure_07_known_event_validation_by_variant.csv", index=False)

    lons = gridded["lon"].values
    lats = gridded["lat"].values
    Lon, Lat = np.meshgrid(lons, lats)
    gridded_df = pd.DataFrame(
        {"lat": Lat.flatten(), "lon": Lon.flatten(), "gridded_vulnerability": gridded.values.flatten()})
    atomic_to_csv(gridded_df, out_dir / "figure_07_gridded_vulnerability.csv", index=False)

    log_sections = {
        "Overview": {
            "sapei_variable": sapei_var, "scdhi_variable": scdhi_var,
            "smoothing_months": SMOOTH_MONTHS, "event_baseline": baseline,
            "period_start": str(data["period_start"].date()),
            "period_end": str(data["period_end"].date()),
        },
        "Panel definitions": (
            "A: gridded climate stress composite (STI heat, -SAPEI drought, "
            "-SCDHI compound, averaged across event months). "
            "B: regional EVI, mean season- and trend-adjusted demand Delta_z across the three events "
            "per climate region -- see Note on averaging below. "
            "C: risk surface, System_Risk = (-SCDHI) x (statewide demand z-score). "
            "D: EMPIRICAL return-period plotting positions of System_Risk (descriptive only: quote the "
            "fitted values in 'Return period' below, with their intervals)."
        ),
        "Event definition (shared across Figures 2 and 4-7)": data["event_meta"],
        "Region abbreviation key (caption text)": format_region_key(data["regions_order"]),
        "Regional event effects (EVI components, FDR-corrected)": comp.to_string(index=False),
        **inference["log_sections"],
        "Return period of the observed maximum (fitted, with uncertainty)": rpa["summary"],
        "Return period of the same month's climate hazard on the full record":
            rpa["hazard"] if rpa["hazard"] is not None else "no full-record hazard series available",
        "POT threshold stability": rpa["stability"].to_string(index=False),
        "Risk-index sensitivity (variants vs the manuscript product)": sensitivity.to_string(index=False),
        "Double-calm months in the manuscript product": calm,
        "Validation against documented grid emergencies": validation_df.to_string(index=False),
        "Validation summary": validation_summary,
        "Validation by index variant (does an index without the climate term also flag them?)":
            validation_by_variant.to_string(index=False),
        "Key numbers": {
            "max_system_risk": float(risk_series.max()),
            "max_system_risk_date": str(risk_series.idxmax().date()),
            "max_return_period_years_EMPIRICAL": float(rp_df["Return_Period_Years"].iloc[0]),
            "pct_months_positive_risk": float(100 * (risk_series > 0).mean()),
            "evi_range": [float(evi.min()), float(evi.max())],
            "highest_evi_region": str(evi.idxmax()),
        },
        "Inference": INFERENCE_NOTE,
        "Note": (
            "EVI averages Delta_z across Heatwave/Drought/Compound into one per-region score; if events "
            "pull in opposite directions for a region, the average can partly cancel and understate real "
            "vulnerability, the per-event, FDR-corrected q-values in the composites CSV are the more "
            "granular reference."
        ),
    }
    write_figure_log(out_dir / "figure_07_systemic_vulnerability", log_sections)
    print(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
