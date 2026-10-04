"""
Ingest EIA-861M state-level monthly electricity sales workbooks (two
separate files, different eras) into a combined statewide monthly demand
series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_HEADER_SCAN_ROWS = 5


def _build_combined_column_names(df_raw: pd.DataFrame, header_row_idx: int) -> list:
    """Combine the sector-group row (forward-filled) with the sub-column-type row."""
    sector_row = df_raw.iloc[header_row_idx - 2].replace("", np.nan).ffill()
    subcol_row = df_raw.iloc[header_row_idx - 1]
    labels_row = df_raw.iloc[header_row_idx]

    columns = []
    for i in range(len(labels_row)):
        if i < 4:
            columns.append(str(labels_row.iloc[i]).strip())
        else:
            columns.append(f"{str(sector_row.iloc[i]).strip()}_{str(subcol_row.iloc[i]).strip()}")
    return columns


def load_raw_demand_sheet(path, sheet_name: str, state: str = "CA") -> pd.Series:
    """Read one EIA-861M sheet and return that state's TOTAL_Sales as a monthly Series (MWh)."""
    df_raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    header_row_idx = None
    for i in range(min(MAX_HEADER_SCAN_ROWS, len(df_raw))):
        if "Year" in set(df_raw.iloc[i].astype(str)):
            header_row_idx = i
            break
    if header_row_idx is None:
        raise ValueError(f"could not locate a header row containing 'Year' in sheet '{sheet_name}'")

    columns = _build_combined_column_names(df_raw, header_row_idx)
    data = df_raw.iloc[header_row_idx + 1:].copy()
    data.columns = columns

    valid_year = pd.to_numeric(data["Year"], errors="coerce")
    data = data[valid_year.notna()].copy()
    data["Year"] = valid_year[valid_year.notna()].astype(int)
    data["Month"] = pd.to_numeric(data["Month"], errors="coerce").astype(int)
    data = data[data["State"].astype(str).str.strip() == state].copy()

    total_col = "TOTAL_Sales"
    if total_col not in data.columns:
        raise ValueError(f"expected column '{total_col}' not found; got {list(data.columns)}")

    data["Date"] = pd.to_datetime(data["Year"].astype(str) + "-" + data["Month"].astype(str) + "-01")
    data[total_col] = pd.to_numeric(data[total_col], errors="coerce")

    return data.set_index("Date")[total_col].sort_index()


def combine_demand_eras(old_series: pd.Series, new_series: pd.Series) -> pd.Series:
    """Combine the two era series, preferring the newer file wherever both cover the same month."""
    combined = pd.concat([old_series, new_series])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()
