from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["Year", "Month", "CountyName", "Sector", "Consumption (MWh)"]
MAX_PREAMBLE_ROWS_TO_SCAN = 5


def _find_header_row(df_raw: pd.DataFrame):
    for i in range(min(MAX_PREAMBLE_ROWS_TO_SCAN, len(df_raw))):
        row_values = {str(v).strip() for v in df_raw.iloc[i]}
        if all(col in row_values for col in REQUIRED_COLUMNS):
            return i
    return None

def load_raw_consumption(path, sheet_name="Electricity Consumption") -> pd.DataFrame:
    df_raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_row_idx = _find_header_row(df_raw)
    if header_row_idx is None:
        raise ValueError(
            f"could not find a header row containing {REQUIRED_COLUMNS} "
            f"within the first {MAX_PREAMBLE_ROWS_TO_SCAN} rows"
        )

    new_header = [str(v).strip() for v in df_raw.iloc[header_row_idx]]
    data = df_raw.iloc[header_row_idx + 1:].copy()
    data.columns = new_header
    data = data.reset_index(drop=True)

    data["Date"] = pd.to_datetime(
        data["Year"].astype(int).astype(str) + "-" + data["Month"].astype(int).astype(str) + "-01"
    )
    data["CountyName"] = data["CountyName"].astype(str).str.upper().str.strip()
    data["Consumption (MWh)"] = data["Consumption (MWh)"].astype(float)

    return data[["Date", "CountyName", "Sector", "Consumption (MWh)"]]
