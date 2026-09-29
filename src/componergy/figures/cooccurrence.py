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
