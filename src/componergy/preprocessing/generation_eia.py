from __future__ import annotations

import pandas as pd

SOURCE_GROUPS = {
    "Fossil": ["Coal", "Natural Gas", "Petroleum", "Other Gases"],
    "Hydro": ["Hydroelectric Conventional"],
    "Solar": ["Solar Thermal and Photovoltaic"],
    "Wind": ["Wind"],
    "Nuclear": ["Nuclear"],
    "Other Renewable": ["Geothermal", "Wood and Wood Derived Fuels", "Other Biomass"],
}


def load_raw_sheets(path, state="CA", producer_type="Total Electric Power Industry") -> pd.DataFrame:
    all_sheets = pd.read_excel(path, sheet_name=None)
    frames = [sheet for sheet in all_sheets.values() if "ENERGY SOURCE" in sheet.columns]
    combined = pd.concat(frames, ignore_index=True)
    return combined[
        (combined["STATE"] == state) & (combined["TYPE OF PRODUCER"] == producer_type)
        ].copy()
