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


def plot_event_choropleth(ax, regions_plot, event_name, comp, state_outline, cmap_obj, norm):
    sub = comp[comp["Event"] == event_name].set_index("Region")
    g = regions_plot.copy()
    g["Delta_pct"] = g["Region"].map(sub["Delta_pct"].to_dict())
    g["q_value"] = g["Region"].map(sub["q_value"].to_dict())

    g.plot(ax=ax, column="Delta_pct", cmap=cmap_obj, norm=norm, linewidth=1.0, edgecolor="white")
    state_outline.plot(ax=ax, color="black", linewidth=1.1)

    g_sig = g[g["q_value"] < Q_ALPHA_FDR]
    if len(g_sig):
        g_sig.boundary.plot(ax=ax, color="black", linewidth=2.0)
    g_marg = g[(g["q_value"] >= Q_ALPHA_FDR) & (g["q_value"] < Q_ALPHA_FDR_MARGINAL)]
    if len(g_marg):
        g_marg.boundary.plot(ax=ax, color="#555555", linewidth=1.6)

    for _, r in g.iterrows():
        if r["centroid"] is None or r["centroid"].is_empty:
            continue
        tcol = text_color_for_bg(r["Delta_pct"], norm, cmap_obj, light_thresh=0.55)
        stroke_col = "black" if tcol == "white" else "white"
        ax.text(r["centroid"].x, r["centroid"].y, r["abbr"], ha="center", va="center",
                fontsize=14, fontweight="bold", color=tcol, alpha=0.98,
                path_effects=[pe.withStroke(linewidth=1.6, foreground=stroke_col), pe.Normal()])

    ax.set_title(event_name, pad=6)
    ax.set_axis_off()


def plot_bubble_matrix(ax, comp: pd.DataFrame, regions_order, events_order, cmap_bub="RdBu_r"):
    mat = comp.pivot(index="Region", columns="Event", values="Delta_z").reindex(index=regions_order,
                                                                                columns=events_order)
    qmat = comp.pivot(index="Region", columns="Event", values="q_value").reindex(index=regions_order,
                                                                                 columns=events_order)

    max_abs = np.nanmax(np.abs(mat.values)) if np.isfinite(mat.values).any() else 1.0
    max_abs = max(0.25, float(max_abs))
    size_min, size_max = 50, 900

    y_spacing = 1.15
    ypos = (np.arange(len(regions_order))[::-1]) * y_spacing
    region_to_y = {r: ypos[i] for i, r in enumerate(regions_order)}

    norm_bub = Normalize(vmin=-max_abs, vmax=max_abs)

    for r in regions_order:
        for j, ev in enumerate(events_order):
            val = mat.loc[r, ev]
            q = qmat.loc[r, ev]
            if pd.isna(val):
                continue

            s = size_min + (np.abs(val) / max_abs) ** 1.35 * (size_max - size_min)

            if pd.notna(q) and q < Q_ALPHA_FDR:
                ec, lw = "black", 1.6
            elif pd.notna(q) and q < Q_ALPHA_FDR_MARGINAL:
                ec, lw = "#444444", 1.2
            else:
                ec, lw = "none", 0.0

            ax.scatter(j, region_to_y[r], s=s, c=[val], cmap=cmap_bub, norm=norm_bub,
                       edgecolors=ec, linewidths=lw, alpha=0.95)

    ax.set_yticks(list(region_to_y.values()))
    ax.set_yticklabels(regions_order)
    ax.set_xticks(np.arange(len(events_order)))
    ax.set_xticklabels(events_order)
    ax.set_xlim(-0.6, len(events_order) - 0.4)

    y_min = -1.0 * y_spacing
    y_max = (len(regions_order) - 1) * y_spacing + 1.0 * y_spacing
    ax.set_ylim(y_min, y_max)

    ax.grid(True, axis="both", alpha=0.15, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return norm_bub, max_abs, size_min, size_max


def build_zonal_figure(comp: pd.DataFrame, regions_gdf, county_boundary_dissolved, regions_order,
                       events_order=("Heatwave", "Drought", "Compound")):
    regions_plot = regions_gdf.copy()
    regions_plot["abbr"] = regions_plot["Region"].map(region_abbrev)
    regions_plot["centroid"] = regions_plot.geometry.representative_point()

    delta_pct_all = comp["Delta_pct"].replace([np.inf, -np.inf], np.nan).dropna().values
    vmax_pct = np.nanpercentile(np.abs(delta_pct_all), 90) if len(delta_pct_all) else 10.0
    vmax_pct = max(5.0, float(vmax_pct))

    fig = plt.figure(figsize=(14.8, 10.2))
    gs = GridSpec(nrows=2, ncols=4, figure=fig, height_ratios=[1.0, 1.35], width_ratios=[1, 1, 1, 0.10])

    map_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax_map = fig.add_subplot(gs[0, 3])

    cmap_map_obj = plt.get_cmap("RdBu_r")
    norm_map = Normalize(vmin=-vmax_pct, vmax=vmax_pct)

    for ax, ev in zip(map_axes, events_order):
        plot_event_choropleth(ax, regions_plot, ev, comp, county_boundary_dissolved, cmap_map_obj, norm_map)

    sm_map = ScalarMappable(norm=norm_map, cmap=cmap_map_obj)
    sm_map.set_array([])
    cb_map = fig.colorbar(sm_map, cax=cax_map, orientation="vertical")
    cb_map.set_label("$\\Delta$% consumption", labelpad=8)

    ax_bub = fig.add_subplot(gs[1, :])
    norm_bub, max_abs, size_min, size_max = plot_bubble_matrix(ax_bub, comp, regions_order, list(events_order))

    cax2 = ax_bub.inset_axes([1.01, 0.10, 0.02, 0.80])
    sm_bub = ScalarMappable(norm=norm_bub, cmap="RdBu_r")
    sm_bub.set_array([])
    cb_bub = fig.colorbar(sm_bub, cax=cax2, orientation="vertical")
    cb_bub.set_label("$\\Delta z$ consumption", labelpad=10)

    legend_vals = [v for v in [0.25, 0.50, 1.00] if v <= max_abs + 1e-9]
    handles = []
    for v in legend_vals:
        s = size_min + (np.abs(v) / max_abs) ** 1.35 * (size_max - size_min)
        handles.append(ax_bub.scatter([], [], s=s, c="none", edgecolors="#333333", linewidths=1.0))
    ax_bub.legend(handles, [f"|$\\Delta z$| = {v:.2f}" for v in legend_vals], title="Effect size",
                  frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                  labelspacing=1.6, handletextpad=0.8, borderaxespad=0.2)

    fig.tight_layout(rect=[0.0, 0.0, 0.98, 0.98])
    return fig


def main(sapei_var: str = "sapei_3m", scdhi_var: str = "scdhi_3m") -> None:
    from componergy.download.climate_regions import get_california_climate_regions
    from componergy.download.boundaries import get_california_counties
    from componergy.analysis.regions import build_county_to_region_map
    from componergy.preprocessing.consumption_cec import load_raw_consumption
    from componergy.netcdf_io import atomic_to_csv
    from componergy.paths import INDICES_FILE, COUNTY_CONSUMPTION_XLSX, FIGURES_DIR

    out_dir = FIGURES_DIR / "figure_04"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading climate regions, counties, and consumption data...")
    regions_gdf = get_california_climate_regions()
    counties_gdf = get_california_counties()
    county_region_map = build_county_to_region_map(counties_gdf, regions_gdf)
    consumption_long_df = load_raw_consumption(COUNTY_CONSUMPTION_XLSX)
    indices_ds = xr.open_dataset(INDICES_FILE)
    county_boundary_dissolved = counties_gdf.dissolve().boundary

    print("preparing zonal data...")
    data = prepare_zonal_data(indices_ds, consumption_long_df, regions_gdf, county_region_map,
                              sapei_var=sapei_var, scdhi_var=scdhi_var)

    print("computing composites...")
    comp = compute_zonal_composites(data)

    print("saving figure...")
    fig = build_zonal_figure(comp, regions_gdf, county_boundary_dissolved, data["regions_order"])
    fig.savefig(out_dir / "figure_04_zonal_consumption.jpg", bbox_inches="tight")
    plt.close(fig)

    atomic_to_csv(comp, out_dir / "figure_04_event_composites.csv", index=False)

    log_sections = {
        "Overview": {
            "sapei_variable": sapei_var, "scdhi_variable": scdhi_var,
            "n_regions": len(data["regions_order"]),
            "period_start": str(data["period_start"].date()),
            "period_end": str(data["period_end"].date()),
        },
        "Event counts (statewide classification)": {k: int(v.sum()) for k, v in data["event_masks"].items()},
        "Normal period count": int(data["normal"].sum()),
        "Event composites (FDR-corrected)": comp.to_string(index=False),
        "Note": (
            "Significance markers (map outline thickness, bubble edge) use "
            "Benjamini-Hochberg FDR-corrected q-values across all region x "
            "event tests, not raw p-values, given the number of comparisons "
            "involved (regions x event types)."
        ),
    }
    write_figure_log(out_dir / "figure_04_zonal_consumption", log_sections)
    print(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
