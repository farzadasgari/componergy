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


def compute_sector_composites(data: dict) -> pd.DataFrame:
    """Event-vs-normal composite per sector, in Delta_GWh/Delta_pct/Delta_z, FDR-corrected."""
    z_table = compute_composite_table(data["sector_z"], data["event_masks"], data["normal"])
    raw_table = compute_composite_table(data["sector_raw"], data["event_masks"], data["normal"])

    merged = z_table.merge(
        raw_table[["Event", "Source", "Mean_event", "Mean_normal"]],
        on=["Event", "Source"], suffixes=("_z", "_raw"),
    )
    merged["Delta_GWh"] = merged["Mean_event_raw"] - merged["Mean_normal_raw"]
    merged["Delta_pct"] = 100.0 * merged["Delta_GWh"] / merged["Mean_normal_raw"]
    merged["q_value"] = benjamini_hochberg_fdr(merged["p_value"].values)

    return merged.rename(columns={"Source": "Sector", "Difference": "Delta_z"})


def build_sector_wheel_figure(comp: pd.DataFrame, sectors_order, events_order=("Heatwave", "Drought", "Compound")):
    """Polar wheel: one angular wedge per sector, one concentric ring per event."""
    mat_g = comp.pivot(index="Sector", columns="Event", values="Delta_GWh").reindex(index=sectors_order,
                                                                                    columns=events_order)
    mat_z = comp.pivot(index="Sector", columns="Event", values="Delta_z").reindex(index=sectors_order,
                                                                                  columns=events_order)
    mat_q = comp.pivot(index="Sector", columns="Event", values="q_value").reindex(index=sectors_order,
                                                                                  columns=events_order)

    abs_g = np.abs(mat_g.values)
    gmax = float(np.nanpercentile(abs_g[np.isfinite(abs_g)], 95)) if np.isfinite(abs_g).any() else 1.0
    gmax = max(1e-6, gmax)

    vals_z = mat_z.values
    vmax_z = float(np.nanpercentile(np.abs(vals_z[np.isfinite(vals_z)]), 90)) if np.isfinite(vals_z).any() else 1.0
    vmax_z = max(0.5, vmax_z)

    cmap_obj = plt.get_cmap("RdBu_r")
    normz = Normalize(vmin=-vmax_z, vmax=vmax_z)

    n_sec = len(sectors_order)
    theta = np.linspace(0, 2 * np.pi, n_sec, endpoint=False)
    width = (2 * np.pi / n_sec) * 0.88

    ring_gap, ring_thick_max, ring_base = 0.12, 0.55, 0.80
    ring_start = {
        "Heatwave": ring_base,
        "Drought": ring_base + (ring_thick_max + ring_gap),
        "Compound": ring_base + 2 * (ring_thick_max + ring_gap),
    }

    fig = plt.figure(figsize=(13.8, 9.0))
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[1.22, 0.24], figure=fig)
    ax = fig.add_subplot(gs[0, 0], projection="polar")
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(alpha=0.12, linewidth=0.8)
    ax.spines["polar"].set_visible(False)

    for ev in events_order:
        base = ring_start[ev]
        for i, sec in enumerate(sectors_order):
            dg, dz, q = mat_g.loc[sec, ev], mat_z.loc[sec, ev], mat_q.loc[sec, ev]
            if pd.isna(dg) or pd.isna(dz):
                continue

            thick = (min(abs(dg) / gmax, 1.0) ** 0.85) * ring_thick_max
            face = cmap_obj(normz(dz))

            if pd.notna(q) and q < Q_ALPHA_FDR:
                ec, lw = "black", 1.4
            elif pd.notna(q) and q < Q_ALPHA_FDR_MARGINAL:
                ec, lw = "#444444", 1.1
            else:
                ec, lw = "none", 0.0

            ax.bar(theta[i], thick, width=width, bottom=base, color=face,
                   edgecolor=ec, linewidth=lw, align="edge", alpha=0.97)

    r_outer_ring = ring_start["Compound"] + ring_thick_max
    outer = r_outer_ring + 0.06
    ax.set_ylim(0, outer + 0.10)

    for i, sec in enumerate(sectors_order):
        lab = SECTOR_ABBREV.get(sec, str(sec)[:7])
        ang = theta[i] + width / 2
        ha = "right" if 90 < np.degrees(ang) < 270 else "left"
        ax.text(ang, outer, lab, fontsize=14, fontweight="bold", ha=ha, va="center", clip_on=False)

    for ev in events_order:
        ax.text(np.deg2rad(2), ring_start[ev] + ring_thick_max * 0.55, ev, fontsize=16,
                ha="left", va="center", color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    totals = comp.groupby("Event")["Delta_GWh"].sum().reindex(events_order)
    center_txt = "Total $\\Delta$GWh vs normal\n" + "\n".join([f"{ev}: {totals[ev]:+.1f}" for ev in events_order])
    ax.text(0.5, 0.5, center_txt, transform=ax.transAxes, ha="center", va="center", fontsize=14)

    ref = np.nanpercentile(abs_g[np.isfinite(abs_g)], [50, 80, 95]) if np.isfinite(abs_g).any() else [1, 2, 3]
    ref = np.unique(np.round(ref, 1))
    ax_leg.text(0.02, 0.97, "Contribution magnitude", transform=ax_leg.transAxes, fontsize=14, va="top")

    y = 0.88
    for r in ref[:3]:
        thick = (min(r / gmax, 1.0) ** 0.85) * ring_thick_max
        ax_leg.add_patch(plt.Rectangle((0.06, y - 0.018), 0.25, 0.036, transform=ax_leg.transAxes,
                                       facecolor="#dddddd", edgecolor="#333333", linewidth=1.0))
        ax_leg.add_patch(plt.Rectangle((0.06, y - 0.018), 0.25, 0.036 * (thick / ring_thick_max),
                                       transform=ax_leg.transAxes, facecolor="#777777", edgecolor="none"))
        ax_leg.text(0.34, y, f"|$\\Delta$GWh| $\\approx$ {r:.1f}", transform=ax_leg.transAxes, va="center", fontsize=14)
        y -= 0.075

    y -= 0.03
    ax_leg.text(0.02, y, "Significance (FDR-corrected)", transform=ax_leg.transAxes, fontsize=14, va="top")
    y -= 0.09
    ax_leg.add_patch(plt.Rectangle((0.06, y - 0.018), 0.10, 0.036, transform=ax_leg.transAxes,
                                   facecolor="white", edgecolor="black", linewidth=1.4))
    ax_leg.text(0.20, y, "q < 0.05", transform=ax_leg.transAxes, va="center", fontsize=13)
    y -= 0.075
    ax_leg.add_patch(plt.Rectangle((0.06, y - 0.018), 0.10, 0.036, transform=ax_leg.transAxes,
                                   facecolor="white", edgecolor="#444444", linewidth=1.1))
    ax_leg.text(0.20, y, "0.05 $\\leq$ q < 0.10", transform=ax_leg.transAxes, va="center", fontsize=13)

    sm = ScalarMappable(norm=normz, cmap="RdBu_r")
    sm.set_array([])
    cax = ax_leg.inset_axes([0.08, 0.03, 0.18, 0.20])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("$\\Delta z$ sector anomaly", labelpad=8, fontsize=16)
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(width=0.8, length=3, labelsize=14)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.05, wspace=0.02)
    return fig
