"""
Ingest EIA-923 "generation_monthly.xlsx" (state-level monthly generation by
energy source) into a single-state, source-grouped monthly table.

The raw file has one sheet per year range plus a notes sheet; rows cover
all states, five producer-type categories, and a per-state "Total"
energy-source row alongside the individual fuel sources (including
Pumped Storage, which can be negative -- net of pumping consumption).

The Total computed here (sum of the six SOURCE_GROUPS) intentionally
excludes Pumped Storage and the small residual "Other" category, since
Pumped Storage reinjects previously-generated energy rather than
producing it. Excluded_MWh reports exactly how much that leaves out per
month, against EIA's own reported total, as a running reconciliation
check rather than a silent omission.
"""

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
    """
    Read every data sheet in the workbook (skipping any without an
    ENERGY SOURCE column, e.g. the notes sheet), concatenate, and filter
    to one state and one producer-type category.

    producer_type defaults to "Total Electric Power Industry", which is
    itself the sum across the other producer-type categories in the raw
    file -- using it avoids double-counting generation across categories.
    """
    all_sheets = pd.read_excel(path, sheet_name=None)
    frames = [sheet for sheet in all_sheets.values() if "ENERGY SOURCE" in sheet.columns]
    combined = pd.concat(frames, ignore_index=True)
    return combined[
        (combined["STATE"] == state) & (combined["TYPE OF PRODUCER"] == producer_type)
        ].copy()


def pivot_by_source_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-format rows (one per state/month/energy source) into a
    wide monthly table with the six SOURCE_GROUPS as columns.

    Returns
    -------
    pd.DataFrame
        Indexed by month-start Date, with SOURCE_GROUPS columns, Total
        (their sum), EIA_Reported_Total (the source file's own "Total"
        row for that month, NaN if absent), and Excluded_MWh
        (EIA_Reported_Total - Total; see module docstring).
    """
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


def main(state: str = "CA", force: bool = False) -> None:
    """
    Read EIA_GENERATION_RAW_FILE, group by source, and write GENERATION_MONTHLY_FILE.

    Idempotent: skips entirely if the output already exists, unless force=True.
    """
    from componergy.netcdf_io import atomic_to_csv
    from componergy.paths import EIA_GENERATION_RAW_FILE, GENERATION_MONTHLY_FILE, ensure_dirs

    ensure_dirs()

    if GENERATION_MONTHLY_FILE.exists() and not force:
        print(f"{GENERATION_MONTHLY_FILE} already exists, skipping (pass force=True to rebuild).")
        return

    print(f"reading {EIA_GENERATION_RAW_FILE} (all sheets)...")
    raw = load_raw_sheets(EIA_GENERATION_RAW_FILE, state=state)
    print(f"found {len(raw)} rows for state={state}")

    grouped = pivot_by_source_group(raw)

    max_excluded_frac = (grouped["Excluded_MWh"].abs() / grouped["EIA_Reported_Total"].abs()).max()
    print(f"max |excluded (pumped storage + other)| / reported total across all months: {max_excluded_frac:.4f}")

    atomic_to_csv(grouped, GENERATION_MONTHLY_FILE)
    print(f"wrote {GENERATION_MONTHLY_FILE}: {len(grouped)} months")


if __name__ == "__main__":
    main()
