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
