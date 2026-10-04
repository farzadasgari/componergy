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
