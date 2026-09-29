"""
Supply-demand imbalance figure: statewide Demand - Generation over time,
event-composite mean shifts, distributional comparison, and a bootstrap
Risk Attributable Fraction (RAF) forest plot -- how much more likely the
imbalance is to exceed a normal-period risk threshold during each event
type.
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
from componergy.analysis.events import classify_heatwave, classify_drought, classify_compound
from componergy.analysis.block_bootstrap import block_bootstrap_raf
from componergy.analysis.superadditivity import superadditivity_test
from componergy.analysis.placebo import placebo_table
from componergy.analysis.system_boundary import validate_system_boundary, compute_import_fraction

SMOOTH_MONTHS = 3
RISK_Q = 0.90
N_BOOT = 5000
BOOT_SEED = 42
EPS = 1e-9

C_HEAT = "crimson"
C_DROUGHT = "deepskyblue"
C_COMP = "purple"
C_IMB = "#222222"
C_GEN = "#d95f02"
C_DEM = "#1f78b4"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
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
                           sapei_var="sapei_3m", scdhi_var="scdhi_3m", smooth_months=SMOOTH_MONTHS):
    """Align statewide demand, generation, and climate signals; classify events."""
    gen_gwh = generation_df["Total"] / 1000.0
    dem_gwh = demand_df["total_sales"] / 1000.0

    weights = compute_area_weights(indices_ds["lat"])
    sti = zscore(compute_statewide_mean(indices_ds["sti"], weights).to_series())
    sapei = zscore(compute_statewide_mean(indices_ds[sapei_var], weights).to_series())
    scdhi = zscore(compute_statewide_mean(indices_ds[scdhi_var], weights).to_series())
    sti = sti.rolling(smooth_months, center=True, min_periods=1).mean()
    sapei = sapei.rolling(smooth_months, center=True, min_periods=1).mean()
    scdhi = scdhi.rolling(smooth_months, center=True, min_periods=1).mean()

    common = gen_gwh.index.intersection(dem_gwh.index).intersection(sti.index).intersection(sapei.index).intersection(
        scdhi.index)

    gen_gwh = gen_gwh.loc[common].rolling(smooth_months, center=True, min_periods=1).mean()
    dem_gwh = dem_gwh.loc[common].rolling(smooth_months, center=True, min_periods=1).mean()
    sti, sapei, scdhi = sti.loc[common], sapei.loc[common], scdhi.loc[common]

    imb_gwh = dem_gwh - gen_gwh
    imb_z = zscore(imb_gwh)

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
        "common": common,
        "gen_gwh": gen_gwh, "dem_gwh": dem_gwh, "imb_gwh": imb_gwh, "imb_z": imb_z,
        "event_masks": event_masks, "normal": normal,
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


def compute_imbalance_composites(data: dict, risk_q=RISK_Q, n_boot=N_BOOT, seed=BOOT_SEED):
    """Event-vs-normal composite (Delta_GWh/Delta_pct/Delta_z, FDR-corrected) plus bootstrap RAF, per event."""
    imb_gwh = data["imb_gwh"]
    normal = data["normal"]

    thr_risk = np.nanquantile(imb_gwh[normal.values], risk_q) if normal.sum() > 5 else np.nan

    z_table = compute_composite_table(imb_gwh.to_frame("Imbalance").apply(zscore), data["event_masks"], normal)
    raw_table = compute_composite_table(imb_gwh.to_frame("Imbalance"), data["event_masks"], normal)
    merged = z_table.merge(
        raw_table[["Event", "Source", "Mean_event", "Mean_normal"]],
        on=["Event", "Source"], suffixes=("_z", "_raw"),
    )
    merged["Delta_GWh"] = merged["Mean_event_raw"] - merged["Mean_normal_raw"]
    merged["Delta_pct"] = 100.0 * merged["Delta_GWh"] / merged["Mean_normal_raw"]
    merged["q_value"] = benjamini_hochberg_fdr(merged["p_value"].values)

    x_nm = imb_gwh[normal.values].dropna().values
    raf_rows = []
    for i, event_name in enumerate(sorted(data["event_masks"])):
        mask = data["event_masks"][event_name]
        x_ev = imb_gwh[mask.values].dropna().values
        raf, lo, hi, p_e, p_n = bootstrap_raf(x_ev, x_nm, thr_risk, n_boot=n_boot, seed=seed + i)

        block_res = block_bootstrap_raf(
            imb_gwh.values, mask.reindex(imb_gwh.index).fillna(False).values,
            normal.reindex(imb_gwh.index).fillna(False).values,
            thr_risk, n_boot=n_boot, seed=seed + i,
        )

        raf_rows.append({
            "Event": event_name, "RiskThreshold_GWh_q90_normal": thr_risk,
            "P_imbalance_gt_thr_event": p_e, "P_imbalance_gt_thr_normal": p_n,
            "RAF": raf, "RAF_CI95_lo": lo, "RAF_CI95_hi": hi,
            "RAF_block_CI95_lo": block_res["ci_lo"], "RAF_block_CI95_hi": block_res["ci_hi"],
            "RAF_block_length_months": block_res["block"],
        })
    raf_df = pd.DataFrame(raf_rows)

    result = merged.rename(columns={"Source": "Metric", "Difference": "Delta_z"}).merge(raf_df, on="Event")
    return result.drop(columns=["Metric"])
