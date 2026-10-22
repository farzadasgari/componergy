"""
Supply-demand imbalance figure: statewide Demand - Generation over time,
event-composite mean shifts, distributional comparison, and a bootstrap
Risk Attributable Fraction (RAF) forest plot
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
from componergy.analysis.events import build_event_masks
from componergy.analysis.block_bootstrap import block_bootstrap_raf
from componergy.analysis.regression import (
    adjusted_composite_table, normal_baseline_anomalies, shift_range, DEFAULT_MIN_SHIFT,
)
from componergy.analysis.demand_crosscheck import compare_demand_sources
from componergy.analysis.system_boundary import validate_system_boundary, compute_import_fraction
from componergy.figures.composite_outputs import write_composite_inference, INFERENCE_NOTE
from componergy.figures.print_style import SERIF_FONTS, EVENT_STYLE

SMOOTH_MONTHS = 3
TREND = "linear"
RISK_Q = 0.90
N_BOOT = 5000
BOOT_SEED = 42
EPS = 1e-9

C_HEAT = EVENT_STYLE["Heatwave"]["color"]
C_DROUGHT = EVENT_STYLE["Drought"]["color"]
C_COMP = EVENT_STYLE["Compound"]["color"]
C_IMB = "#222222"
C_GEN = "#d95f02"
C_DEM = "#1f78b4"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": SERIF_FONTS,
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
                           sapei_var="sapei_3m", scdhi_var="scdhi_3m", smooth_months=SMOOTH_MONTHS,
                           baseline="full_record"):
    """Align statewide demand, generation, and climate signals; classify events."""
    gen_gwh = generation_df["Total"] / 1000.0
    dem_gwh = demand_df["total_sales"] / 1000.0

    weights = compute_area_weights(indices_ds["lat"])
    sti = zscore(compute_statewide_mean(indices_ds["sti"], weights).to_series())
    sapei = zscore(compute_statewide_mean(indices_ds[sapei_var], weights).to_series())
    scdhi = zscore(compute_statewide_mean(indices_ds[scdhi_var], weights).to_series())
    sti_full, sapei_full, scdhi_full = sti, sapei, scdhi

    common = gen_gwh.index.intersection(dem_gwh.index).intersection(sti.index).intersection(sapei.index).intersection(
        scdhi.index)

    gen_gwh = gen_gwh.loc[common].rolling(smooth_months, center=True, min_periods=1).mean()
    dem_gwh = dem_gwh.loc[common].rolling(smooth_months, center=True, min_periods=1).mean()

    imb_gwh = dem_gwh - gen_gwh
    imb_z = zscore(imb_gwh)

    event_masks, normal, event_meta, event_flags = build_event_masks(
        sti_full, sapei_full, scdhi_full, window_index=common,
        smooth_months=smooth_months, baseline=baseline, return_flags=True,
    )
    imb_adj, imb_baseline = normal_baseline_anomalies(imb_gwh, normal)

    return {
        "common": common,
        "gen_gwh": gen_gwh, "dem_gwh": dem_gwh, "imb_gwh": imb_gwh, "imb_z": imb_z,
        "imb_adj": imb_adj, "imb_baseline": imb_baseline,
        "event_masks": event_masks, "normal": normal,
        "event_meta": event_meta, "event_flags": event_flags,
    }


def bootstrap_raf(values_event, values_normal, thr, n_boot=N_BOOT, seed=BOOT_SEED):
    """Risk Attributable Fraction: P(value > thr | event) / P(value > thr | normal) - 1, with a bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    e = np.asarray(values_event, dtype=float)
    n = np.asarray(values_normal, dtype=float)
    e = e[np.isfinite(e)]
    n = n[np.isfinite(n)]
    if len(e) < 8 or len(n) < 8:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    p_e = np.mean(e > thr)
    p_n = np.mean(n > thr)
    raf = p_e / max(p_n, EPS) - 1.0

    boots = np.empty(n_boot)
    for i in range(n_boot):
        eb = rng.choice(e, size=len(e), replace=True)
        nb = rng.choice(n, size=len(n), replace=True)
        pe = np.mean(eb > thr)
        pn = np.mean(nb > thr)
        boots[i] = pe / max(pn, EPS) - 1.0
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return raf, lo, hi, p_e, p_n


def _raf_stat(values, event_mask, normal_mask, thr):
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    e, n = v[event_mask & ok], v[normal_mask & ok]
    if len(e) < 3 or len(n) < 3:
        return np.nan
    return float(np.mean(e > thr) / max(np.mean(n > thr), EPS) - 1.0)


def raf_shift_test(values, event_mask, normal_mask, thr, min_shift=DEFAULT_MIN_SHIFT):
    """Is RAF larger than event timing alone produces? Calendar-shift randomization."""
    v = np.asarray(values, dtype=float)
    ev, nm = np.asarray(event_mask, dtype=bool), np.asarray(normal_mask, dtype=bool)
    other = ~(ev | nm)
    observed = _raf_stat(v, ev, nm, thr)
    if not np.isfinite(observed):
        return {"observed": np.nan, "p_upper": np.nan, "p_two_sided": np.nan, "n_shifts": 0}
    null = np.array([_raf_stat(v, np.roll(ev, int(s)), ~np.roll(ev, int(s)) & ~other, thr)
                     for s in shift_range(len(v), min_shift)])
    null = null[np.isfinite(null)]
    S = len(null)
    return {"observed": observed, "n_shifts": int(S),
            "p_upper": float((1 + np.sum(null >= observed)) / (S + 1)),
            "p_two_sided": float((1 + np.sum(np.abs(null - null.mean()) >= abs(observed - null.mean()))) / (S + 1))}


def compute_imbalance_composites(data: dict, risk_q=RISK_Q, n_boot=N_BOOT, seed=BOOT_SEED, trend=TREND):
    """Season- and trend-adjusted event effect on the imbalance, plus RAF, per event."""
    imb_gwh, normal = data["imb_gwh"], data["normal"]
    imb_adj = data.get("imb_adj")
    if imb_adj is None:
        imb_adj, _ = normal_baseline_anomalies(imb_gwh, normal, trend=trend)

    adj = adjusted_composite_table(imb_gwh.to_frame("Imbalance"), data["event_masks"], normal, trend=trend)
    adj["Delta_GWh"] = adj["Delta_native"]
    adj["q_value"] = benjamini_hochberg_fdr(adj["p_value"].values)

    def risk_threshold(series):
        return np.nanquantile(series[normal.values], risk_q) if normal.sum() > 5 else np.nan

    thr_adj, thr_raw = risk_threshold(imb_adj), risk_threshold(imb_gwh)
    x_nm_adj = imb_adj[normal.values].dropna().values
    x_nm_raw = imb_gwh[normal.values].dropna().values

    raf_rows = []
    for i, event_name in enumerate(sorted(data["event_masks"])):
        mask = data["event_masks"][event_name]
        ev_arr = mask.reindex(imb_gwh.index).fillna(False).values
        nm_arr = normal.reindex(imb_gwh.index).fillna(False).values

        raf, lo, hi, p_e, p_n = bootstrap_raf(imb_adj[mask.values].dropna().values, x_nm_adj, thr_adj,
                                              n_boot=n_boot, seed=seed + i)
        raf_u, lo_u, hi_u, _, _ = bootstrap_raf(imb_gwh[mask.values].dropna().values, x_nm_raw, thr_raw,
                                                n_boot=n_boot, seed=seed + i)
        block_res = block_bootstrap_raf(imb_adj.values, ev_arr, nm_arr, thr_adj, n_boot=n_boot, seed=seed + i)
        shift = raf_shift_test(imb_adj.values, ev_arr, nm_arr, thr_adj)

        raf_rows.append({
            "Event": event_name, "RiskThreshold_adjusted_anomaly_GWh": thr_adj,
            "P_imbalance_gt_thr_event": p_e, "P_imbalance_gt_thr_normal": p_n,
            "RAF": raf, "RAF_CI95_lo": lo, "RAF_CI95_hi": hi,
            "RAF_block_CI95_lo": block_res["ci_lo"], "RAF_block_CI95_hi": block_res["ci_hi"],
            "RAF_block_length_months": block_res["block"],
            "RAF_p_shift_upper": shift["p_upper"], "RAF_p_shift_two_sided": shift["p_two_sided"],
            "RAF_n_shifts": shift["n_shifts"],
            "RAF_unadjusted": raf_u, "RAF_unadjusted_CI95_lo": lo_u, "RAF_unadjusted_CI95_hi": hi_u,
        })
    raf_df = pd.DataFrame(raf_rows)

    result = adj.rename(columns={"Source": "Metric"}).merge(raf_df, on="Event")
    return result.drop(columns=["Metric"])


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def add_letter(ax, letter, x=0.01, y=0.98, fs=18):
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top", fontsize=fs)


def plot_signal_panel(ax, data):
    common = data["common"]
    ax.plot(common, zscore(data["dem_gwh"]), color=C_DEM, lw=2.1, alpha=0.85, label="Demand")
    ax.plot(common, zscore(data["gen_gwh"]), color=C_GEN, lw=2.1, alpha=0.85, label="Generation")
    ax.plot(common, data["imb_z"], color=C_IMB, lw=2.9, alpha=0.95, label="Imbalance")
    ax.axhline(0, color="k", lw=1.0, alpha=0.35, zorder=0)

    ax.set_ylabel("Standardized anomaly")
    ax.set_xlabel("Time (Year)")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.tick_params(axis="x", which="major", length=6, width=1.2)
    ax.tick_params(axis="x", which="minor", length=3, width=0.9)

    style_axis(ax)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.03, 0.05), borderaxespad=0.0)


def plot_delta_bar_panel(ax, comp: pd.DataFrame, order=("Heatwave", "Drought", "Compound")):
    event_colors = {"Heatwave": C_HEAT, "Drought": C_DROUGHT, "Compound": C_COMP}
    vals = comp.set_index("Event").loc[list(order), "Delta_GWh"].to_numpy(dtype=float)
    cols = [event_colors[e] for e in order]
    x = np.arange(len(order))

    ax.bar(x, vals, color=cols, alpha=0.90, width=0.70, edgecolor="black", linewidth=1.4)
    for patch, e in zip(ax.patches, order):
        patch.set_hatch(EVENT_STYLE[e]["hatch"])
    ax.axhline(0, color="k", lw=1.0, alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylabel("$\\Delta$ Imbalance vs normal (GWh)")
    style_axis(ax)


def plot_distribution_panel(ax, data, comp: pd.DataFrame, order=("Heatwave", "Drought", "Compound")):
    event_colors = {"Heatwave": C_HEAT, "Drought": C_DROUGHT, "Compound": C_COMP}
    imb_gwh = data.get("imb_adj", data["imb_gwh"])

    x_all = imb_gwh[np.isfinite(imb_gwh.values)].values
    xmin, xmax = np.nanpercentile(x_all, [2, 98])
    grid = np.linspace(xmin, xmax, 260)

    x_nm = imb_gwh[data["normal"].values].dropna().values
    if len(x_nm) > 10:
        ax.plot(grid, gaussian_kde(x_nm)(grid), color="#6f6f6f", lw=2.2, ls="--", label="Normal")

    for ev in order:
        m = data["event_masks"][ev].values
        x_ev = imb_gwh[m].dropna().values
        if len(x_ev) > 10:
            ax.plot(grid, gaussian_kde(x_ev)(grid), color=event_colors[ev], lw=2.2, alpha=0.95, label=ev)

    thr_col = "RiskThreshold_adjusted_anomaly_GWh"
    thr_risk = comp[thr_col].iloc[0] if thr_col in comp.columns else np.nan
    if np.isfinite(thr_risk):
        ax.axvline(thr_risk, color="k", lw=1.0, alpha=0.55)

    ax.set_ylabel("Probability density")
    ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(0.05, 1.1))
    style_axis(ax)


def plot_raf_panel(ax, comp: pd.DataFrame, order=("Heatwave", "Drought", "Compound")):
    """RAF forest plot. Reference line at RAF=0 (no amplification)."""
    event_colors = {"Heatwave": C_HEAT, "Drought": C_DROUGHT, "Compound": C_COMP}
    tmp = comp.set_index("Event").loc[list(order)]
    raf = tmp["RAF"].to_numpy(dtype=float)
    lo = tmp["RAF_CI95_lo"].to_numpy(dtype=float)
    hi = tmp["RAF_CI95_hi"].to_numpy(dtype=float)

    y = np.arange(len(order))
    xerr = np.vstack([raf - lo, hi - raf])

    ax.errorbar(raf, y, xerr=xerr, fmt="o", ms=7.5, lw=1.8, capsize=4, color="#222222", zorder=2)
    for i, ev in enumerate(order):
        ax.scatter(raf[i], i, s=95, color=event_colors[ev], zorder=3)

    ax.axvline(0.0, color="k", lw=1.0, ls="--", alpha=0.6, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("RAF")
    ax.set_ylim(-0.6, len(order) - 0.4)

    finite_lo = lo[np.isfinite(lo)]
    finite_hi = hi[np.isfinite(hi)]
    if len(finite_lo) and len(finite_hi):
        lo_min, hi_max = float(np.nanmin(finite_lo)), float(np.nanmax(finite_hi))
        pad = 0.12 * (hi_max - lo_min + EPS)
        ax.set_xlim(lo_min - pad, hi_max + pad)

    style_axis(ax)


def build_imbalance_figure(data: dict, comp: pd.DataFrame):
    fig = plt.figure(figsize=(16.5, 9.4))
    gs = GridSpec(nrows=2, ncols=3, figure=fig, height_ratios=[1.25, 1.0], wspace=0.34, hspace=0.34)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.97, bottom=0.10)

    axA = fig.add_subplot(gs[0, :])
    plot_signal_panel(axA, data)
    add_letter(axA, "A")

    axB = fig.add_subplot(gs[1, 0])
    plot_delta_bar_panel(axB, comp)
    add_letter(axB, "B")

    axC = fig.add_subplot(gs[1, 1])
    plot_distribution_panel(axC, data, comp)
    add_letter(axC, "C")

    axD = fig.add_subplot(gs[1, 2])
    plot_raf_panel(axD, comp)
    add_letter(axD, "D")

    return fig


DEMAND_SOURCES = ("cec", "eia861m")


def load_demand_sources() -> dict:
    """Statewide monthly demand (MWh) from each source that is present on disk; None if absent."""

    from componergy.paths import COUNTY_CONSUMPTION_XLSX, DEMAND_MONTHLY_FILE
    from componergy.preprocessing.consumption_cec import load_raw_consumption, statewide_monthly_mwh

    out = {"cec": None, "eia861m": None}
    if COUNTY_CONSUMPTION_XLSX.exists():
        out["cec"] = statewide_monthly_mwh(load_raw_consumption(COUNTY_CONSUMPTION_XLSX))
    if DEMAND_MONTHLY_FILE.exists():
        d = pd.read_csv(DEMAND_MONTHLY_FILE, parse_dates=[0], index_col=0)
        d.index = pd.DatetimeIndex(d.index.values, name=None)
        out["eia861m"] = d["total_sales"]
    return out


def main(sapei_var: str = "sapei_3m", scdhi_var: str = "scdhi_3m", baseline: str = "full_record",
         demand_source: str = "cec") -> None:
    """Build the imbalance figure, its CSVs, and the log/json."""
    from componergy.netcdf_io import atomic_to_csv
    from componergy.paths import INDICES_FILE, GENERATION_MONTHLY_FILE, FIGURES_DIR

    if demand_source not in DEMAND_SOURCES:
        raise ValueError(f"demand_source must be one of {DEMAND_SOURCES}, got {demand_source!r}")

    out_dir = FIGURES_DIR / "figure_06"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading generation, demand, and climate data...")
    generation_df = pd.read_csv(GENERATION_MONTHLY_FILE, parse_dates=[0], index_col=0)
    generation_df.index = pd.DatetimeIndex(generation_df.index.values, name=None)
    sources = load_demand_sources()
    if sources[demand_source] is None:
        raise FileNotFoundError(
            f"demand source {demand_source!r} is not available on disk. 'cec' needs the restricted CEC "
            "county consumption workbook (paths.COUNTY_CONSUMPTION_XLSX); 'eia861m' needs "
            "paths.DEMAND_MONTHLY_FILE (run scripts/run_demand_pipeline.py). Use --demand-source to choose."
        )
    indices_ds = xr.open_dataset(INDICES_FILE)
    alt_name = [s for s in DEMAND_SOURCES if s != demand_source][0]

    print(f"preparing imbalance data (demand source: {demand_source}, baseline: {baseline})...")
    data = prepare_imbalance_data(indices_ds, generation_df, sources[demand_source].to_frame("total_sales"),
                                  sapei_var=sapei_var, scdhi_var=scdhi_var, baseline=baseline)

    print("computing composites and RAF...")
    comp = compute_imbalance_composites(data)

    print("saving figure...")
    fig = build_imbalance_figure(data, comp)
    fig.savefig(out_dir / "figure_06_imbalance.jpg", bbox_inches="tight")
    plt.close(fig)

    atomic_to_csv(comp, out_dir / "figure_06_event_metrics.csv", index=False)

    print("computing amplification, robustness and placebo tables...")
    inference = write_composite_inference(data["imb_gwh"].to_frame("Imbalance"), data["event_masks"],
                                          data["normal"], out_dir, "figure_06", label="Metric", trend=TREND)

    alt_comp, alt_period = None, None
    if sources[alt_name] is not None:
        alt_data = prepare_imbalance_data(indices_ds, generation_df, sources[alt_name].to_frame("total_sales"),
                                          sapei_var=sapei_var, scdhi_var=scdhi_var, baseline=baseline)
        alt_comp = compute_imbalance_composites(alt_data)
        alt_period = f"{alt_data['common'].min().date()} to {alt_data['common'].max().date()}"
        atomic_to_csv(alt_comp, out_dir / f"figure_06_event_metrics_{alt_name}_demand.csv", index=False)

    demand_check = (compare_demand_sources(sources["cec"], sources["eia861m"])
                    if sources["cec"] is not None and sources["eia861m"] is not None
                    else "only one demand source is available, so no cross-check is possible")

    gen_gwh_raw = generation_df["Total"] / 1000.0
    boundary = {name: validate_system_boundary(series / 1000.0, gen_gwh_raw)
                for name, series in sources.items() if series is not None}
    atomic_to_csv(compute_import_fraction(sources[demand_source] / 1000.0, gen_gwh_raw),
                  out_dir / "figure_06_system_boundary.csv")

    metrics_df = pd.DataFrame({
        "Demand_GWh": data["dem_gwh"], "Generation_GWh": data["gen_gwh"],
        "Imbalance_GWh": data["imb_gwh"], "Imbalance_adjusted_GWh": data["imb_adj"],
        "Imbalance_z": data["imb_z"],
    })
    for name, mask in data["event_masks"].items():
        metrics_df[f"{name}"] = mask.reindex(metrics_df.index).fillna(False)
    metrics_df["Normal"] = data["normal"].reindex(metrics_df.index).fillna(False)
    atomic_to_csv(metrics_df, out_dir / "figure_06_imbalance_metrics.csv")

    log_sections = {
        "Overview": {
            "sapei_variable": sapei_var, "scdhi_variable": scdhi_var,
            "smoothing_months": SMOOTH_MONTHS, "event_baseline": baseline,
            "demand_source": demand_source,
            "period_start": str(data["common"].min().date()),
            "period_end": str(data["common"].max().date()),
            "n_months": len(data["common"]),
        },
        "Imbalance definition": (
            "Demand_GWh - Generation_GWh. Positive = demand exceeds in-state generation. This is a "
            "monthly energy-accounting gap: it includes net imports, but also storage and transmission "
            "losses and any boundary mismatch between the demand and generation series, none of which "
            "is measured here. Describe it as such, not as grid reliability."
        ),
        "Demand source": {
            "primary": demand_source,
            "why": ("CEC is the current default, pending a decision on the primary source; EIA-861M "
                    "is computed alongside it as a cross-check."),
            "alternative_source": alt_name if alt_comp is not None else "not available",
            "alternative_period": alt_period,
        },
        "Demand source cross-check (CEC vs EIA-861M)": demand_check,
        "System boundary validation (per demand source)": boundary,
        "Event definition (shared across Figures 2 and 4-7)": data["event_meta"],
        "Event effects and RAF (FDR-corrected)": comp.to_string(index=False),
        **inference["log_sections"],
        "Results with the alternative demand source": (
            alt_comp.to_string(index=False) if alt_comp is not None else "alternative source not available"),
        "Inference": INFERENCE_NOTE,
        "Note": (
            "Delta_pct divides by the counterfactual baseline mean imbalance and can be extreme if that "
            "baseline is near zero; Delta_GWh and Delta_z are the more robust metrics for this figure. "
            "RAF is computed on the imbalance as a seasonally adjusted counterfactual anomaly; "
            "RAF_unadjusted reproduces the original raw-imbalance version, which is confounded by the "
            "season (summer imbalances exceed a normal-months threshold largely regardless of events). "
            "RAF_p_shift_upper is the calendar-shift p-value for RAF > 0 and cannot fall below "
            "1/(RAF_n_shifts + 1); the RAF bootstrap CIs are NOT calibrated when events form a few "
            "episodes and are kept for reference. The bootstrap uses a fixed, deterministic per-event "
            "seed offset, not a hash of the event name -- Python's hash() on strings is randomized "
            "per-process (PYTHONHASHSEED) and would give different intervals on every re-run."
        ),
    }
    write_figure_log(out_dir / "figure_06_imbalance", log_sections)
    print(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
