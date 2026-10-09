"""
CDHE co-occurrence frequency by climate region and electricity demand sector.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

S1_SECTORS = {
    "Residential": "Residential",
    "Commercial": "Commercial",
    "Industrial": "Industrial",
    "Agricultural": "Ag With Water Pump",
}

DEMAND_HIGH_Q = 0.90

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "dejavuserif",
    "savefig.dpi": 600,
})

import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def aggregate_consumption_to_region_sector(consumption_long_df: pd.DataFrame,
                                           county_region_map: pd.DataFrame) -> pd.DataFrame:
    """Monthly consumption in GWh with a (Region, Sector) MultiIndex on the columns."""
    merged = consumption_long_df.merge(county_region_map, on="CountyName", how="left")

    unmapped = sorted(merged.loc[merged["Region"].isna(), "CountyName"].unique().tolist())
    if unmapped:
        raise ValueError(f"Counties missing region assignment: {unmapped}")

    return (
            merged.groupby(["Date", "Region", "Sector"])["Consumption (MWh)"]
            .sum()
            .unstack(["Region", "Sector"])
            .sort_index()
            / 1000.0
    )


def compute_cooccurrence_frequency(regional_scdhi: pd.DataFrame, region_sector_gwh: pd.DataFrame,
                                   sectors=None, demand_high_q=DEMAND_HIGH_Q,
                                   smooth_months=3) -> pd.DataFrame:
    """Fraction of each region's CDHE months that coincide with elevated demand, per sector."""
    from componergy.figures.composite_stats import deseasonalize_monthly, zscore, series_to_data_array
    from componergy.analysis.events import classify_compound

    if sectors is None:
        sectors = S1_SECTORS

    rows = []
    for region in regional_scdhi.columns:
        scdhi = zscore(regional_scdhi[region].rolling(smooth_months, center=True, min_periods=1).mean())
        cdhe = classify_compound(series_to_data_array(scdhi)).to_series()
        cdhe = cdhe.reindex(region_sector_gwh.index).fillna(False).astype(bool)
        n_cdhe = int(cdhe.sum())

        for label, cec_name in sectors.items():
            key = (region, cec_name)
            if key not in region_sector_gwh.columns:
                rows.append({"Region": region, "Sector": label, "N_CDHE_months": n_cdhe,
                             "N_concurrent": 0, "Cooccurrence_Frequency": np.nan})
                continue

            demand = region_sector_gwh[key]
            demand_sm = demand.rolling(smooth_months, center=True, min_periods=1).mean()
            demand_z = zscore(deseasonalize_monthly(demand_sm))

            aligned = demand_z.reindex(cdhe.index)
            thr = float(np.nanquantile(aligned.values, demand_high_q))
            high_demand = aligned > thr

            concurrent = int((cdhe & high_demand.fillna(False)).sum())
            freq = float(concurrent / n_cdhe) if n_cdhe > 0 else np.nan

            rows.append({"Region": region, "Sector": label, "N_CDHE_months": n_cdhe,
                         "N_concurrent": concurrent, "Cooccurrence_Frequency": freq})

    return pd.DataFrame(rows)


def plot_cooccurrence_heatmap(freq_df: pd.DataFrame, sectors=None, figsize=(11, 8)):
    """Region x sector heatmap with the frequency printed in each cell."""
    if sectors is None:
        sectors = S1_SECTORS
    sector_order = list(sectors.keys())

    mat = freq_df.pivot(index="Region", columns="Sector", values="Cooccurrence_Frequency")
    mat = mat.reindex(columns=sector_order)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(mat.values, cmap="YlOrRd", vmin=0.0, aspect="auto")

    ax.set_xticks(np.arange(len(sector_order)))
    ax.set_xticklabels(sector_order, fontsize=13)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=13)

    vmax = np.nanmax(mat.values) if np.isfinite(mat.values).any() else 1.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=11, color="#777777")
                continue
            color = "white" if val > 0.6 * vmax else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=12, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("CDHE co-occurrence frequency", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.set_xlabel("Electricity demand sector", fontsize=15)
    ax.set_ylabel("Climate region", fontsize=15)
    return fig


def main(scdhi_var: str = "scdhi_3m") -> None:
    """Build S1 plus its frequency CSV and log."""
    from componergy.download.climate_regions import get_california_climate_regions
    from componergy.download.boundaries import get_california_counties
    from componergy.analysis.regions import build_county_to_region_map, compute_regional_mean
    from componergy.preprocessing.consumption_cec import load_raw_consumption
    from componergy.netcdf_io import atomic_to_csv
    from componergy.paths import INDICES_FILE, COUNTY_CONSUMPTION_XLSX, FIGURES_DIR
    from componergy.figures.stats_log import write_figure_log

    out_dir = FIGURES_DIR / "supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading regions, counties, consumption, and climate data...")
    regions_gdf = get_california_climate_regions()
    counties_gdf = get_california_counties()
    county_region_map = build_county_to_region_map(counties_gdf, regions_gdf)
    consumption_long_df = load_raw_consumption(COUNTY_CONSUMPTION_XLSX)
    ds = xr.open_dataset(INDICES_FILE)

    print("computing regional SCDHI and region-sector consumption...")
    regional_scdhi = compute_regional_mean(ds[scdhi_var], regions_gdf)
    region_sector_gwh = aggregate_consumption_to_region_sector(consumption_long_df, county_region_map)

    common = regional_scdhi.index.intersection(region_sector_gwh.index)
    # regional_scdhi is deliberately NOT trimmed: event thresholds use the full climate record
    region_sector_gwh = region_sector_gwh.loc[common]

    print("computing co-occurrence frequencies...")
    freq_df = compute_cooccurrence_frequency(regional_scdhi, region_sector_gwh)
    atomic_to_csv(freq_df, out_dir / "figure_S1_cooccurrence_frequency.csv", index=False)

    print("building S1...")
    fig = plot_cooccurrence_heatmap(freq_df)
    fig.savefig(out_dir / "figure_S1_cooccurrence.jpg", bbox_inches="tight")
    plt.close(fig)

    valid = freq_df["Cooccurrence_Frequency"].dropna()
    log_sections = {
        "Overview": {
            "scdhi_variable": scdhi_var,
            "demand_high_quantile": DEMAND_HIGH_Q,
            "n_regions": int(freq_df["Region"].nunique()),
            "n_sectors": int(freq_df["Sector"].nunique()),
            "period_start": str(common.min().date()),
            "period_end": str(common.max().date()),
        },
        "Co-occurrence frequency by region and sector": freq_df.to_string(index=False),
        "Summary": {
            "mean_frequency": float(valid.mean()) if len(valid) else np.nan,
            "max_frequency": float(valid.max()) if len(valid) else np.nan,
            "expected_by_chance": float(1.0 - DEMAND_HIGH_Q),
        },
        "Note": (
            "Frequency is defined per the S1 caption as the fraction of that "
            "region's CDHE months during which the given sector also shows "
            "elevated demand, where 'elevated' is the top decile of that "
            "region-sector's deseasonalized, z-scored demand."
        ),
    }
    write_figure_log(out_dir / "supplementary_cooccurrence", log_sections)
    print(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
