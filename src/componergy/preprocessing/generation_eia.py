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


def pivot_by_source_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(
        df["YEAR"].astype(int).astype(str) + "-" + df["MONTH"].astype(int).astype(str) + "-01"
    )

    reported_total = (
        df[df["ENERGY SOURCE"] == "Total"]
        .set_index("Date")["GENERATION (Megawatthours)"]
    )

    wide = (
        df[df["ENERGY SOURCE"] != "Total"]
        .pivot_table(index="Date", columns="ENERGY SOURCE", values="GENERATION (Megawatthours)", aggfunc="sum")
        .fillna(0.0)
    )

    grouped = pd.DataFrame(index=wide.index)
    for group_name, source_list in SOURCE_GROUPS.items():
        grouped[group_name] = sum(wide.get(s, 0.0) for s in source_list)

    grouped["Total"] = grouped[list(SOURCE_GROUPS)].sum(axis=1)
    grouped["EIA_Reported_Total"] = reported_total.reindex(grouped.index)
    grouped["Excluded_MWh"] = grouped["EIA_Reported_Total"] - grouped["Total"]

    return grouped.sort_index()
