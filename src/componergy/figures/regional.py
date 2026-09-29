"""
Regional generation-climate-electricity figure: heatmaps of standardized
anomalies by climate region over time, and a slope-bar summary of each
region's dominant climate driver.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from componergy.analysis.regions import build_county_to_region_map, compute_regional_mean
from componergy.figures.print_style import SOURCE_STYLE
from componergy.figures.composite_stats import best_lag_slope, benjamini_hochberg_fdr, compute_incremental_r2
from componergy.figures.stats_log import write_figure_log

SMOOTH_MONTHS = 3
LAGS = range(-6, 7)
DRIVER_COLORS = {"STI": "crimson", "SAPEI": "deepskyblue", "SCDHI": "purple"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "axes.linewidth": 1.2,
    "savefig.dpi": 600,
})

import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not std or not np.isfinite(std):
        return series * 0.0
    return (series - series.mean()) / std


def aggregate_consumption_to_regional(consumption_long_df: pd.DataFrame,
                                      county_region_map: pd.DataFrame) -> pd.DataFrame:
    """Sum county consumption (all sectors, all planning-area splits) to monthly regional totals, in GWh."""
    merged = consumption_long_df.merge(county_region_map, on="CountyName", how="left")

    unmapped = sorted(merged.loc[merged["Region"].isna(), "CountyName"].unique().tolist())
    if unmapped:
        raise ValueError(f"Counties missing region assignment: {unmapped}")

    regional = (
            merged.groupby(["Date", "Region"])["Consumption (MWh)"]
            .sum()
            .unstack("Region")
            .sort_index()
            / 1000.0
    )
    return regional


def prepare_data(indices_ds: xr.Dataset, consumption_long_df: pd.DataFrame, regions_gdf,
                 county_region_map: pd.DataFrame,
                 sapei_var="sapei_3m", scdhi_var="scdhi_3m"):
    """
    Align regional electricity consumption with regional climate signals.

    Returns a dict with regions_order (list) and four smoothed,
    z-scored, common-period DataFrames (time index, one column per
    region): electricity_z, sti_z, sapei_z, scdhi_z, plus period_start/end.
    """
    electricity_regional = aggregate_consumption_to_regional(consumption_long_df, county_region_map)

    sti_regional = compute_regional_mean(indices_ds["sti"], regions_gdf)
    sapei_regional = compute_regional_mean(indices_ds[sapei_var], regions_gdf)
    scdhi_regional = compute_regional_mean(indices_ds[scdhi_var], regions_gdf)

    regions_order = [r for r in regions_gdf["Region"] if r in electricity_regional.columns]

    start = max(electricity_regional.index.min(), sti_regional.index.min(),
                sapei_regional.index.min(), scdhi_regional.index.min())
    end = min(electricity_regional.index.max(), sti_regional.index.max(),
              sapei_regional.index.max(), scdhi_regional.index.max())

    electricity_regional = electricity_regional.loc[start:end, regions_order]
    sti_regional = sti_regional.loc[start:end, regions_order]
    sapei_regional = sapei_regional.loc[start:end, regions_order]
    scdhi_regional = scdhi_regional.loc[start:end, regions_order]

    def smooth_and_z(df):
        smoothed = df.rolling(SMOOTH_MONTHS, center=True, min_periods=1).mean()
        return smoothed.apply(zscore, axis=0)

    return {
        "regions_order": regions_order,
        "electricity_z": smooth_and_z(electricity_regional),
        "sti_z": smooth_and_z(sti_regional),
        "sapei_z": smooth_and_z(sapei_regional),
        "scdhi_z": smooth_and_z(scdhi_regional),
        "period_start": start,
        "period_end": end,
    }


def plot_heatmap(ax, data, regions_order, row_label, vlim=2.5, cmap="viridis"):
    """
    One heatmap panel: regions on the y-axis, time on the x-axis.

    NaN cells (a region with no overlapping grid cells for this
    variable, see analysis.regions.compute_regional_mean) render as a
    distinct light gray rather than the colormap's own out-of-range
    color or the page's white background, so missing data reads as
    missing rather than as a value or blank space.
    """
    mat = data[regions_order].T.values
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#dddddd")
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=-vlim, vmax=vlim, cmap=cmap_obj)
    ax.set_yticks(np.arange(len(regions_order)))
    ax.set_yticklabels(regions_order)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(1.02, 0.5, row_label, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=16)
    return im


def build_heatmap_figure(data):
    regions_order = data["regions_order"]
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(13, 1.1 * len(regions_order) + 6), sharex=True)
    fig.subplots_adjust(left=0.16, right=0.83, top=0.96, bottom=0.06, hspace=0.15)

    im0 = plot_heatmap(axes[0], data["sti_z"], regions_order, "STI")
    plot_heatmap(axes[1], data["sapei_z"], regions_order, "SAPEI")
    plot_heatmap(axes[2], data["scdhi_z"], regions_order, "SCDHI")
    plot_heatmap(axes[3], data["electricity_z"], regions_order, "Electricity")

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Time (Year)")

    tick_idx = np.arange(0, len(data["electricity_z"].index), 24)
    tick_labels = [data["electricity_z"].index[i].strftime("%Y") for i in tick_idx]
    for ax in axes:
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels)

    cbar = fig.colorbar(im0, ax=axes, orientation="vertical", fraction=0.05, pad=0.03, shrink=0.9)
    cbar.set_label("Standardized anomaly (z-score)", labelpad=14)

    return fig


def compute_dominance_and_lag_summary(data):
    """
    Best-lag driver regression per (region, driver), with FDR-corrected q-values.

    One best_lag_slope() call per (region, driver) pair; the resulting
    p-values are corrected together across the whole table via
    benjamini_hochberg_fdr(), not read individually against 0.05.
    """
    regions_order = data["regions_order"]
    drivers = {"STI": data["sti_z"], "SAPEI": data["sapei_z"], "SCDHI": data["scdhi_z"]}

    rows = []
    for region in regions_order:
        y = data["electricity_z"][region].dropna()
        for driver_name, driver_df in drivers.items():
            x = driver_df[region].reindex(y.index)
            best = best_lag_slope(x, y, lags=LAGS, min_n=24)
            if best is None:
                continue
            rows.append({"Region": region, "Driver": driver_name, **best})

    summary = pd.DataFrame(rows)
    summary["q_value"] = benjamini_hochberg_fdr(summary["p_value"].values)
    return summary


def plot_slope_bars(ax, summary, regions_order, alpha_fdr=0.05):
    """Bar chart of each region's driver slopes, fading bars whose FDR-corrected q-value exceeds alpha_fdr."""
    drivers = ["STI", "SAPEI", "SCDHI"]
    x = np.arange(len(regions_order))
    width = 0.26

    for j, drv in enumerate(drivers):
        sub = summary[summary["Driver"] == drv].set_index("Region").reindex(regions_order)
        vals = sub["slope"].to_numpy(dtype=float)
        sig = (sub["q_value"] < alpha_fdr).to_numpy()

        bars = ax.bar(x + (j - 1) * width, vals, width=width, label=drv, color=DRIVER_COLORS[drv])
        for bar, is_sig in zip(bars, sig):
            if not is_sig:
                bar.set_alpha(0.35)

    ax.axhline(0, linewidth=1.0, color="#444444", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(regions_order, rotation=90, ha="center")
    ax.set_ylabel("Electricity response per unit driver z-score")
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.22), borderaxespad=0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def compute_incremental_r2_by_region(data):
    """
    Per-region incremental R^2 of SCDHI beyond STI, FDR-corrected across regions.

    See composite_stats.compute_incremental_r2 for the nested-model
    F-test this wraps, and why it is not called "dominance analysis".
    """
    rows = []
    for region in data["regions_order"]:
        y = data["electricity_z"][region].dropna()
        sti = data["sti_z"][region].reindex(y.index)
        scdhi = data["scdhi_z"][region].reindex(y.index)

        combined = pd.concat([y.rename("y"), sti.rename("sti"), scdhi.rename("scdhi")], axis=1).dropna()
        if len(combined) < 24:
            continue

        result = compute_incremental_r2(combined["y"], combined["sti"], combined["scdhi"])
        rows.append({"Region": region, **result})

    incremental = pd.DataFrame(rows)
    incremental["q_value"] = benjamini_hochberg_fdr(incremental["p_value"].values)
    return incremental


def compute_sign_consistency(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.assign(sign=lambda df_: np.sign(df_["slope"]))
        .groupby("Driver")["sign"]
        .value_counts()
        .unstack(fill_value=0)
    )


def main(sapei_var: str = "sapei_3m", scdhi_var: str = "scdhi_3m") -> None:
    """Build the heatmap and slope-bar figures, the five summary CSVs, and the log/json.

    Requires COUNTY_CONSUMPTION_XLSX to already exist locally (restricted
    CEC data -- see preprocessing.consumption_cec); the climate-regions
    and county-boundary downloads are cached automatically on first use.
    """
    from componergy.download.climate_regions import get_california_climate_regions
    from componergy.download.boundaries import get_california_counties
    from componergy.preprocessing.consumption_cec import load_raw_consumption
    from componergy.netcdf_io import atomic_to_csv
    from componergy.paths import INDICES_FILE, COUNTY_CONSUMPTION_XLSX, FIGURES_DIR

    out_dir = FIGURES_DIR / "figure_03"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading climate regions, counties, and consumption data...")
    regions_gdf = get_california_climate_regions()
    counties_gdf = get_california_counties()
    county_region_map = build_county_to_region_map(counties_gdf, regions_gdf)
    consumption_long_df = load_raw_consumption(COUNTY_CONSUMPTION_XLSX)
    indices_ds = xr.open_dataset(INDICES_FILE)

    print("preparing regional data...")
    data = prepare_data(indices_ds, consumption_long_df, regions_gdf, county_region_map,
                        sapei_var=sapei_var, scdhi_var=scdhi_var)

    print("computing statistics...")
    summary = compute_dominance_and_lag_summary(data)
    incremental = compute_incremental_r2_by_region(data)
    sign_check = compute_sign_consistency(summary)

    dominance = (
        summary.sort_values("r2", ascending=False)
        .groupby("Region").first()
        .reset_index()[["Region", "Driver", "r2", "q_value", "lag", "slope"]]
    )
    driver_stats = (
        summary.groupby("Driver")
        .agg(mean_slope=("slope", "mean"), median_slope=("slope", "median"), std_slope=("slope", "std"),
             mean_r2=("r2", "mean"), median_r2=("r2", "median"),
             n_significant_fdr=("q_value", lambda q: int((q < 0.05).sum())))
        .reset_index()
    )
    lag_summary = (
        summary.groupby("Driver")
        .agg(mean_lag=("lag", "mean"), median_lag=("lag", "median"), min_lag=("lag", "min"), max_lag=("lag", "max"))
        .reset_index()
    )

    print("saving figures...")
    heatmap_fig = build_heatmap_figure(data)
    heatmap_fig.savefig(out_dir / "figure_03A_heatmaps.jpg", bbox_inches="tight")
    plt.close(heatmap_fig)

    slope_fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(data["regions_order"])), 5.5))
    plot_slope_bars(ax, summary, data["regions_order"])
    slope_fig.savefig(out_dir / "figure_03B_slope.jpg", bbox_inches="tight")
    plt.close(slope_fig)

    atomic_to_csv(dominance, out_dir / "figure_03_dominance.csv", index=False)
    atomic_to_csv(driver_stats, out_dir / "figure_03_driver_stats.csv", index=False)
    atomic_to_csv(lag_summary, out_dir / "figure_03_lag_summary.csv", index=False)
    atomic_to_csv(sign_check, out_dir / "figure_03_sign_consistency.csv")
    atomic_to_csv(incremental, out_dir / "figure_03_incremental_r2.csv", index=False)

    log_sections = {
        "Overview": {
            "sapei_variable": sapei_var, "scdhi_variable": scdhi_var,
            "smoothing_months": SMOOTH_MONTHS,
            "n_regions": len(data["regions_order"]),
            "period_start": str(data["period_start"].date()),
            "period_end": str(data["period_end"].date()),
        },
        "County to region mapping": {
            "method": "majority area overlap (EPSG:3310)",
            "n_counties_mapped": len(county_region_map),
        },
        "Dominant driver per region (highest R2)": dominance.to_string(index=False),
        "Driver overall statistics across regions": driver_stats.to_string(index=False),
        "Lag summary per driver": lag_summary.to_string(index=False),
        "Sign consistency of slopes": sign_check.to_string(),
        "Incremental R2 (SCDHI beyond STI)": incremental.to_string(index=False),
        "Note": (
            "Climate regions are not electrical balancing areas, transmission "
            "zones, or utility planning areas; regional climate-electricity "
            "relationships shown here should not be read as grid-system risk "
            "without separately modeling balancing areas, transmission, and "
            "imports. All p-values are FDR-corrected (Benjamini-Hochberg) "
            "across regions within each driver; q_value < 0.05 is the "
            "significance criterion used for shading in figure."
        ),
    }
    write_figure_log(out_dir / "figure_03_regional", log_sections)
    print(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
