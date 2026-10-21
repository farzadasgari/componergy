from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from componergy.figures.trend_stats import compute_area_weights, compute_statewide_mean
from componergy.figures.composite_stats import (
    series_to_data_array, compute_composite_table, benjamini_hochberg_fdr, zscore, deseasonalize_monthly,
)
from componergy.figures.stats_log import write_figure_log
from componergy.analysis.events import build_event_masks
from componergy.analysis.regression import adjusted_composite_table
from componergy.figures.composite_outputs import write_composite_inference, INFERENCE_NOTE

Q_ALPHA_FDR = 0.05

# Event masks are mutually exclusive in the composites, so Heatwave/Drought are heat-only/drought-only.
EVENT_LABELS = {"Heatwave": "Heat-only", "Drought": "Drought-only", "Compound": "Compound"}
EVENT_COLORS = {"Heatwave": "crimson", "Drought": "deepskyblue", "Compound": "purple"}
EVENT_OFFSET = 0.24
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


def build_sector_effects_figure(comp: pd.DataFrame, sectors_order, events_order=("Heatwave", "Drought", "Compound")):
    """Dot-and-whisker plot: one row per sector, one offset marker per event type, 95% intervals."""
    eff = sector_effect_intervals(comp)
    n_ev = len(events_order)
    offsets = {ev: (k - (n_ev - 1) / 2) * EVENT_OFFSET for k, ev in enumerate(events_order)}

    fig, ax = plt.subplots(figsize=(11.0, 0.95 * len(sectors_order) + 1.6))
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
    ax.set_yticklabels(sectors_order)
    ax.set_ylim(len(sectors_order) - 0.5, -0.5)
    ax.set_xlabel("$\\Delta$% of fitted baseline (95% interval)")
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.8)
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=EVENT_COLORS[ev], marker="o", ms=9, mew=1.8, lw=1.8, label=EVENT_LABELS[ev])
               for ev in events_order]
    handles += [Line2D([], [], color="#444444", marker="o", ms=9, mew=1.8, ls="none", label="q < 0.05"),
                Line2D([], [], color="#444444", mfc="white", marker="o", ms=9, mew=1.8, ls="none",
                       label="q $\\geq$ 0.05")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=False,
              handlelength=1.6, columnspacing=1.2)

    fig.tight_layout()
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
    fig = build_sector_effects_figure(comp, data["sectors_order"])
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
                INFERENCE_NOTE + " Filled markers use q < 0.05 across all sector x event tests "
                                 "(7 sectors x 3 events = 21); whiskers are 95% intervals from the "
                                 "shift-randomization SE, rescaled to % of the fitted baseline."
        ),
    }
    write_figure_log(out_dir / "figure_05_sector_fingerprint", log_sections)
    print(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
