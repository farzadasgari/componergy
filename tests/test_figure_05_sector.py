import numpy as np
import pandas as pd

from componergy.figures.sector import (
    aggregate_consumption_by_sector, prepare_sector_data, compute_sector_composites, SECTOR_ORDER,
)


def test_aggregate_consumption_by_sector_conserves_total():
    idx_rows = [
        {"Date": pd.Timestamp("2008-01-01"), "CountyName": "ALPHA", "Sector": "Residential",
         "Consumption (MWh)": 1000.0},
        {"Date": pd.Timestamp("2008-01-01"), "CountyName": "BETA", "Sector": "Residential", "Consumption (MWh)": 500.0},
        {"Date": pd.Timestamp("2008-01-01"), "CountyName": "ALPHA", "Sector": "Commercial",
         "Consumption (MWh)": 2000.0},
    ]
    long_df = pd.DataFrame(idx_rows)

    sector_gwh = aggregate_consumption_by_sector(long_df)

    raw_total_gwh = long_df["Consumption (MWh)"].sum() / 1000.0
    sector_total_gwh = sector_gwh.loc["2008-01-01"].sum()
    assert abs(raw_total_gwh - sector_total_gwh) < 1e-9
