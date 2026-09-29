"""
Supplementary Figure S1: CDHE co-occurrence frequency by climate region
and electricity demand sector.
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
