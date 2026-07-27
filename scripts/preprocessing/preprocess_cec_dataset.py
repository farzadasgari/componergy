from componergy.paths import (
    CEC_RAW_FILE,
    CEC_MONTHLY_LONG_FILE,
    CEC_MONTHLY_WIDE_FILE,
)
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


SHEET_NAME = "Electricity Consumption"
VALUE_COL = "Consumption (MWh)"

KEY_COLUMNS = [
    "Year",
    "Month",
    "CountyNum",
    "CountyName",
    "Sector",
]

CEC_MONTHLY_LONG_FILE.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(
    CEC_RAW_FILE,
    sheet_name=SHEET_NAME,
    dtype={
        "Year": int,
        "Month": int,
    },
)

df = df[KEY_COLUMNS + [VALUE_COL]].copy()

negative_mask = df[VALUE_COL] < 0
if negative_mask.any():
    print(
        f"Found {negative_mask.sum()} negative values "
        f"({negative_mask.mean():.2%} of rows)"
    )

county_long = (
    df.groupby(KEY_COLUMNS, as_index=False)[VALUE_COL]
    .sum()
    .rename(columns={VALUE_COL: "Consumption_MWh"})
)

county_wide = (
    county_long.pivot_table(
        index=[
            "Year",
            "Month",
            "CountyNum",
            "CountyName",
        ],
        columns="Sector",
        values="Consumption_MWh",
        aggfunc="sum",
    )
    .fillna(0)
    .reset_index()
)

sector_columns = [
    c
    for c in county_wide.columns
    if c not in {
        "Year",
        "Month",
        "CountyNum",
        "CountyName",
    }
]

county_wide["Total_MWh"] = county_wide[sector_columns].sum(axis=1)

county_long.to_csv(CEC_MONTHLY_LONG_FILE, index=False)
county_wide.to_csv(CEC_MONTHLY_WIDE_FILE, index=False)

print("\nSector summary")
print(
    county_long.groupby("Sector")["Consumption_MWh"]
    .agg(["sum", "mean", "count"])
    .round(1)
)
