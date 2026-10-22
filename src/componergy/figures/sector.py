from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from componergy.figures.trend_stats import compute_area_weights, compute_statewide_mean
from componergy.figures.composite_stats import (
    series_to_data_array, compute_composite_table, benjamini_hochberg_fdr, zscore, deseasonalize_monthly,
)
from componergy.figures.stats_log import write_figure_log
from componergy.analysis.events import build_event_masks
from componergy.analysis.regression import adjusted_composite_table
from componergy.figures.composite_outputs import write_composite_inference, INFERENCE_NOTE
from componergy.figures.print_style import SERIF_FONTS

Q_ALPHA_FDR = 0.05
P_ALPHA_NOMINAL = 0.05

EVENT_LABELS = {"Heatwave": "Heat-only", "Drought": "Drought-only", "Compound": "Compound"}
EVENT_COLORS = {"Heatwave": "crimson", "Drought": "deepskyblue", "Compound": "purple"}
EVENT_OFFSET = 0.24
SECTOR_ABBREV = {"Ag With Water Pump": "AgPump", "Streetlighting": "Streetlight"}

RING_GAP, RING_THICK_MAX, RING_BASE = 0.12, 0.55, 0.80
WEDGE_POWER = 0.85
OUTLINE_NOMINAL_COLOR = "#555555"
FIG_W, FIG_H = 15.6, 7.2
LABEL_PT = 12
PANEL_PT = 18
SECTOR_ORDER = ["Residential", "Commercial", "Industrial", "Ag With Water Pump", "TCU", "Mining", "Streetlighting"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": SERIF_FONTS,
    "mathtext.fontset": "dejavuserif",
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
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
                        sapei_var="sapei_3m", scdhi_var="scdhi_3m", smooth_months=3, baseline="full_record"):
    """Classify statewide climate events and prepare statewide sector consumption in raw (GWh) and z-scored form."""
    sector_raw = aggregate_consumption_by_sector(consumption_long_df)
    sectors_order = [s for s in SECTOR_ORDER if s in sector_raw.columns] + \
                    [s for s in sector_raw.columns if s not in SECTOR_ORDER]
    sector_raw = sector_raw[sectors_order]

    weights = compute_area_weights(indices_ds["lat"])
    sti = zscore(compute_statewide_mean(indices_ds["sti"], weights).to_series())
    sapei = zscore(compute_statewide_mean(indices_ds[sapei_var], weights).to_series())
    scdhi = zscore(compute_statewide_mean(indices_ds[scdhi_var], weights).to_series())
    sti_full, sapei_full, scdhi_full = sti, sapei, scdhi

    start = max(sector_raw.index.min(), sti.index.min(), sapei.index.min(), scdhi.index.min())
    end = min(sector_raw.index.max(), sti.index.max(), sapei.index.max(), scdhi.index.max())

    sector_raw = sector_raw.loc[start:end]
    sti, sapei, scdhi = sti.loc[start:end], sapei.loc[start:end], scdhi.loc[start:end]

    sector_sm = sector_raw.rolling(smooth_months, center=True, min_periods=1).mean()
    sector_local = sector_sm.apply(deseasonalize_monthly, axis=0)
    sector_z = sector_local.apply(zscore, axis=0)

    event_masks, normal, event_meta, event_flags = build_event_masks(
        sti_full, sapei_full, scdhi_full, window_index=sti.index,
        smooth_months=smooth_months, baseline=baseline, return_flags=True,
    )

    return {
        "sectors_order": sectors_order,
        "sector_raw": sector_sm,
        "sector_z": sector_z,
        "event_masks": event_masks,
        "normal": normal,
        "event_meta": event_meta,
        "event_flags": event_flags,
        "period_start": start,
        "period_end": end,
    }


def compute_sector_composites(data: dict, trend: str = "linear") -> pd.DataFrame:
    """Event effect per sector (Delta_GWh / Delta_pct / Delta_z), season- and trend-adjusted, FDR-corrected."""
    adj = adjusted_composite_table(data["sector_raw"], data["event_masks"], data["normal"], trend=trend)
    adj["Delta_GWh"] = adj["Delta_native"]
    adj["q_value"] = benjamini_hochberg_fdr(adj["p_value"].values)
    return adj.rename(columns={"Source": "Sector"})


def sector_effect_intervals(comp: pd.DataFrame) -> pd.DataFrame:
    """Delta% of the fitted baseline with its 95% interval, rescaled from the shift-randomization CI."""
    out = comp[["Event", "Sector", "Delta_pct", "q_value"]].copy()
    lo = 100.0 * comp["CI_lo_native"] / comp["Baseline_mean"]
    hi = 100.0 * comp["CI_hi_native"] / comp["Baseline_mean"]
    out["CI_lo_pct"] = np.fmin(lo, hi)
    out["CI_hi_pct"] = np.fmax(lo, hi)
    return out


def wheel_scales(comp: pd.DataFrame) -> dict:
    """Wedge-thickness scale (95th percentile of |Delta GWh|) and symmetric Delta z colour norm."""
    g = np.abs(comp["Delta_GWh"].to_numpy(float))
    g = g[np.isfinite(g)]
    gmax = max(1e-6, float(np.nanpercentile(g, 95))) if g.size else 1.0
    z = np.abs(comp["Delta_z"].to_numpy(float))
    z = z[np.isfinite(z)]
    vmax_z = max(0.5, float(np.nanpercentile(z, 90))) if z.size else 1.0
    return {"gmax": gmax, "vmax_z": vmax_z, "norm": Normalize(vmin=-vmax_z, vmax=vmax_z)}


def wedge_thickness(delta_gwh: float, gmax: float) -> float:
    return (min(abs(delta_gwh) / gmax, 1.0) ** WEDGE_POWER) * RING_THICK_MAX


def wedge_outline(q: float, p_exact: float) -> tuple[str, str, float]:
    """(edge colour, line style, line width): solid black for q < 0.05, dashed grey for nominal p < 0.05 only."""
    if pd.notna(q) and q < Q_ALPHA_FDR:
        return "black", "-", 1.8
    if pd.notna(p_exact) and p_exact < P_ALPHA_NOMINAL:
        return OUTLINE_NOMINAL_COLOR, "--", 1.6
    return "none", "-", 0.0


def draw_sector_wheel(ax, comp: pd.DataFrame, sectors_order, events_order=("Heatwave", "Drought", "Compound")):
    """Polar bars on a polar axis: one ring per event, one wedge per sector (colour Delta z, thickness |Delta GWh|)."""

    def mat(col):
        return comp.pivot(index="Sector", columns="Event", values=col).reindex(index=sectors_order,
                                                                               columns=events_order)

    mat_g, mat_z, mat_q, mat_p = mat("Delta_GWh"), mat("Delta_z"), mat("q_value"), mat("p_exact")
    sc = wheel_scales(comp)
    cmap = plt.get_cmap("RdBu_r")

    n_sec = len(sectors_order)
    theta = np.linspace(0, 2 * np.pi, n_sec, endpoint=False)
    width = (2 * np.pi / n_sec) * 0.88
    ring_start = {ev: RING_BASE + k * (RING_THICK_MAX + RING_GAP) for k, ev in enumerate(events_order)}

    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(alpha=0.12, linewidth=0.8)
    ax.spines["polar"].set_visible(False)

    for ev in events_order:
        for i, sec in enumerate(sectors_order):
            dg, dz = mat_g.loc[sec, ev], mat_z.loc[sec, ev]
            if pd.isna(dg) or pd.isna(dz):
                continue
            ec, ls, lw = wedge_outline(mat_q.loc[sec, ev], mat_p.loc[sec, ev])
            ax.bar(theta[i], wedge_thickness(dg, sc["gmax"]), width=width, bottom=ring_start[ev],
                   color=cmap(sc["norm"](dz)), edgecolor=ec, linestyle=ls, linewidth=lw, align="edge")

    outer = ring_start[events_order[-1]] + RING_THICK_MAX + 0.06
    ax.set_ylim(0, outer + 0.10)

    for i, sec in enumerate(sectors_order):
        ang = theta[i] + width / 2
        sx, cy = np.sin(ang), np.cos(ang)
        ha = "left" if sx > 0.25 else "right" if sx < -0.25 else "center"
        va = "center" if ha != "center" else ("bottom" if cy > 0 else "top")
        ax.text(ang, outer + 0.10, SECTOR_ABBREV.get(sec, str(sec)), fontsize=LABEL_PT, fontweight="bold",
                ha=ha, va=va, clip_on=False)

    for ev in events_order:
        ax.text(np.deg2rad(2), ring_start[ev] + RING_THICK_MAX * 0.55, EVENT_LABELS[ev], fontsize=LABEL_PT,
                ha="left", va="center", color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    return sc


def draw_wheel_legend(fig, ax_leg, comp: pd.DataFrame, sc: dict) -> None:
    """Thickness key, outline key and Delta z colourbar in the legend column (axes-fraction coordinates)."""
    ax_leg.axis("off")
    g = np.abs(comp["Delta_GWh"].to_numpy(float))
    g = g[np.isfinite(g)]
    ref = np.unique(np.round(np.nanpercentile(g, [50, 80, 95]), 1)) if g.size else np.array([1.0])

    ax_leg.text(0.0, 1.0, "Wedge thickness,\n$|\\Delta$GWh|", transform=ax_leg.transAxes, fontsize=LABEL_PT, va="top",
                linespacing=1.1)
    y = 0.87
    for r in ref[:3]:
        frac = wedge_thickness(r, sc["gmax"]) / RING_THICK_MAX
        ax_leg.add_patch(plt.Rectangle((0.0, y - 0.025), 0.20, 0.05, transform=ax_leg.transAxes,
                                       facecolor="#dddddd", edgecolor="#333333", linewidth=1.0))
        ax_leg.add_patch(plt.Rectangle((0.0, y - 0.025), 0.20, 0.05 * frac, transform=ax_leg.transAxes,
                                       facecolor="#777777", edgecolor="none"))
        ax_leg.text(0.25, y, f"{r:.1f} GWh", transform=ax_leg.transAxes, va="center", fontsize=LABEL_PT)
        y -= 0.075

    y -= 0.08
    ax_leg.text(0.0, y, "Outline", transform=ax_leg.transAxes, fontsize=LABEL_PT, va="top")
    y -= 0.075
    for (ec, ls, lw), label in (((("black", "-", 1.8)), "q < 0.05"),
                                (((OUTLINE_NOMINAL_COLOR, "--", 1.6)), "exact p < 0.05,\nq $\\geq$ 0.05")):
        ax_leg.add_patch(plt.Rectangle((0.0, y - 0.025), 0.20, 0.05, transform=ax_leg.transAxes,
                                       facecolor="white", edgecolor=ec, linestyle=ls, linewidth=lw))
        ax_leg.text(0.25, y, label, transform=ax_leg.transAxes, va="center", fontsize=LABEL_PT, linespacing=1.1)
        y -= 0.095

    sm = ScalarMappable(norm=sc["norm"], cmap="RdBu_r")
    sm.set_array([])
    cax = ax_leg.inset_axes([0.0, 0.0, 0.16, 0.26])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("$\\Delta z$ sector anomaly", labelpad=6, fontsize=LABEL_PT)
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(width=0.8, length=3, labelsize=LABEL_PT)


def draw_sector_effects(ax, comp: pd.DataFrame, sectors_order, events_order=("Heatwave", "Drought", "Compound")):
    """Dot-and-whisker plot: one row per sector, one offset marker per event type, 95% intervals."""
    eff = sector_effect_intervals(comp)
    n_ev = len(events_order)
    offsets = {ev: (k - (n_ev - 1) / 2) * EVENT_OFFSET for k, ev in enumerate(events_order)}

    for i in range(len(sectors_order)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f2f2f2", zorder=0, lw=0)
    ax.axvline(0, color="black", lw=1.2, zorder=1)

    for ev in events_order:
        color = EVENT_COLORS[ev]
        sub = eff[eff["Event"] == ev].set_index("Sector").reindex(sectors_order)
        for i, sec in enumerate(sectors_order):
            r = sub.loc[sec]
            if pd.isna(r["Delta_pct"]):
                continue
            y = i + offsets[ev]
            ax.hlines(y, r["CI_lo_pct"], r["CI_hi_pct"], color=color, lw=1.8, zorder=2)
            significant = pd.notna(r["q_value"]) and r["q_value"] < Q_ALPHA_FDR
            ax.plot(r["Delta_pct"], y, marker="o", ms=9, mew=1.8, color=color,
                    mfc=color if significant else "white", ls="none", zorder=3)

    ax.set_yticks(range(len(sectors_order)))
    ax.set_yticklabels(sectors_order, fontsize=LABEL_PT)
    ax.tick_params(axis="x", labelsize=LABEL_PT)
    ax.set_ylim(len(sectors_order) - 0.5, -0.5)
    ax.set_xlabel("$\\Delta$% of fitted baseline (95% interval)", fontsize=LABEL_PT)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.8)
    ax.set_axisbelow(True)

    handles = [Line2D([], [], color=EVENT_COLORS[ev], marker="o", ms=9, mew=1.8, lw=1.8, label=EVENT_LABELS[ev])
               for ev in events_order]
    handles += [Line2D([], [], color="#444444", marker="o", ms=9, mew=1.8, ls="none", label="q < 0.05"),
                Line2D([], [], color="#444444", mfc="white", marker="o", ms=9, mew=1.8, ls="none",
                       label="q $\\geq$ 0.05")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.17), ncol=3, frameon=False,
              handlelength=1.4, columnspacing=1.0, handletextpad=0.5, fontsize=LABEL_PT)


def build_sector_figure(comp: pd.DataFrame, sectors_order, events_order=("Heatwave", "Drought", "Compound")):
    """Two-panel figure: A radial fingerprint with its legend column, B dot-and-whisker effects (% of baseline)."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def rect(x, y, w, h):
        return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]

    ax_wheel = fig.add_axes(rect(0.95, 1.2, 5.3, 5.3), projection="polar")
    ax_leg = fig.add_axes(rect(7.35, 0.7, 2.25, 6.2))
    ax_eff = fig.add_axes(rect(11.75, 1.65, 3.75, 5.2))

    sc = draw_sector_wheel(ax_wheel, comp, sectors_order, events_order)
    draw_wheel_legend(fig, ax_leg, comp, sc)
    draw_sector_effects(ax_eff, comp, sectors_order, events_order)

    fig.text(0.2 / FIG_W, 1 - 0.1 / FIG_H, "A", fontsize=PANEL_PT, fontweight="bold", ha="left", va="top")
    fig.text(9.95 / FIG_W, 1 - 0.1 / FIG_H, "B", fontsize=PANEL_PT, fontweight="bold", ha="left", va="top")
    return fig


def main(sapei_var: str = "sapei_3m", scdhi_var: str = "scdhi_3m", baseline: str = "full_record") -> None:
    """
    Build the sector effects figure, its composite CSV, and the log/json.

    Requires COUNTY_CONSUMPTION_XLSX to already exist locally (restricted
    CEC data).
    """
    from componergy.preprocessing.consumption_cec import load_raw_consumption
    from componergy.netcdf_io import atomic_to_csv
    from componergy.paths import INDICES_FILE, COUNTY_CONSUMPTION_XLSX, FIGURES_DIR

    out_dir = FIGURES_DIR / "figure_05"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading consumption and climate data...")
    consumption_long_df = load_raw_consumption(COUNTY_CONSUMPTION_XLSX)
    indices_ds = xr.open_dataset(INDICES_FILE)

    print("preparing sector data...")
    data = prepare_sector_data(indices_ds, consumption_long_df, sapei_var=sapei_var, scdhi_var=scdhi_var,
                               baseline=baseline)

    print("computing composites...")
    comp = compute_sector_composites(data)

    print("saving figure...")
    fig = build_sector_figure(comp, data["sectors_order"])
    fig.savefig(out_dir / "figure_05_sector_fingerprint.jpg", bbox_inches="tight")
    plt.close(fig)

    atomic_to_csv(comp, out_dir / "figure_05_sector_composites.csv", index=False)

    print("computing amplification, robustness and placebo tables...")
    inference = write_composite_inference(data["sector_raw"], data["event_masks"], data["normal"],
                                          out_dir, "figure_05", label="Sector")

    log_sections = {
        "Overview": {
            "sapei_variable": sapei_var, "scdhi_variable": scdhi_var,
            "n_sectors": len(data["sectors_order"]),
            "period_start": str(data["period_start"].date()),
            "period_end": str(data["period_end"].date()),
        },
        "Event definition (shared across Figures 2 and 4-7)": data["event_meta"],
        "Sector effects (FDR-corrected)": comp.to_string(index=False),
        **inference["log_sections"],
        "Note": (
                INFERENCE_NOTE + " Panel A: wedge colour is Delta z, thickness |Delta GWh|; solid outline q < 0.05, "
                                 "dashed outline nominal exact p < 0.05 with q >= 0.05. Panel B: filled markers use "
                                 "q < 0.05 across all sector x event tests "
                                 "(7 sectors x 3 events = 21); whiskers are 95% intervals from the "
                                 "shift-randomization SE, rescaled to % of the fitted baseline."
        ),
    }
    write_figure_log(out_dir / "figure_05_sector_fingerprint", log_sections)
    print(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
