import pandas as pd
from pathlib import Path
import tempfile

from componergy.preprocessing.generation_eia import load_raw_sheets, pivot_by_source_group, SOURCE_GROUPS


def _make_test_workbook():
    test_dir = Path(tempfile.mkdtemp())
    test_path = test_dir / "gen.xlsx"

    sources = {
        "Coal": 199857, "Natural Gas": 10192494, "Petroleum": 459703, "Other Gases": 97569,
        "Nuclear": 2379998, "Hydroelectric Conventional": 1590096, "Wind": 133423,
        "Solar Thermal and Photovoltaic": 6500, "Wood and Wood Derived Fuels": 313245,
        "Geothermal": 1085733, "Other Biomass": 178497, "Pumped Storage": -36255, "Other": 18027,
    }
    total = sum(sources.values())

    rows = [
        {"YEAR": 2001, "MONTH": 1, "STATE": "CA", "TYPE OF PRODUCER": "Total Electric Power Industry",
         "ENERGY SOURCE": k, "GENERATION (Megawatthours)": v}
        for k, v in sources.items()
    ]
    rows.append({"YEAR": 2001, "MONTH": 1, "STATE": "CA", "TYPE OF PRODUCER": "Total Electric Power Industry",
                 "ENERGY SOURCE": "Total", "GENERATION (Megawatthours)": total})
    rows.append({"YEAR": 2001, "MONTH": 1, "STATE": "TX", "TYPE OF PRODUCER": "Total Electric Power Industry",
                 "ENERGY SOURCE": "Coal", "GENERATION (Megawatthours)": 88888888})
    rows.append({"YEAR": 2001, "MONTH": 1, "STATE": "CA", "TYPE OF PRODUCER": "Electric Generators, Electric Utilities",
                 "ENERGY SOURCE": "Coal", "GENERATION (Megawatthours)": 55555555})

    with pd.ExcelWriter(test_path) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="2001_Final", index=False)
        pd.DataFrame({"Note": ["metadata"]}).to_excel(writer, sheet_name="EnergySource_Notes", index=False)

    return test_path, sources, total


def test_load_raw_sheets_filters_state_and_producer_type():
    path, sources, total = _make_test_workbook()
    raw = load_raw_sheets(path, state="CA")

    assert (raw["STATE"] == "CA").all()
    assert (raw["TYPE OF PRODUCER"] == "Total Electric Power Industry").all()
    assert 88888888 not in raw["GENERATION (Megawatthours)"].values
    assert 55555555 not in raw["GENERATION (Megawatthours)"].values


def test_source_grouping_matches_manual_sum():
    path, sources, total = _make_test_workbook()
    raw = load_raw_sheets(path, state="CA")
    grouped = pivot_by_source_group(raw)
    row = grouped.loc["2001-01-01"]

    expected_fossil = sources["Coal"] + sources["Natural Gas"] + sources["Petroleum"] + sources["Other Gases"]
    expected_other_renew = sources["Geothermal"] + sources["Wood and Wood Derived Fuels"] + sources["Other Biomass"]

    assert abs(row["Fossil"] - expected_fossil) < 1e-6
    assert abs(row["Hydro"] - sources["Hydroelectric Conventional"]) < 1e-6
    assert abs(row["Other Renewable"] - expected_other_renew) < 1e-6


def test_reconciliation_against_eia_reported_total():
    path, sources, total = _make_test_workbook()
    raw = load_raw_sheets(path, state="CA")
    grouped = pivot_by_source_group(raw)
    row = grouped.loc["2001-01-01"]

    assert abs(row["EIA_Reported_Total"] - total) < 1e-6
    expected_excluded = sources["Pumped Storage"] + sources["Other"]
    assert abs(row["Excluded_MWh"] - expected_excluded) < 1e-6


def test_month_with_no_reported_total_gives_nan_not_crash():
    test_dir = Path(tempfile.mkdtemp())
    test_path = test_dir / "gen_no_total.xlsx"
    rows = [
        {"YEAR": 2002, "MONTH": 1, "STATE": "CA", "TYPE OF PRODUCER": "Total Electric Power Industry",
         "ENERGY SOURCE": "Coal", "GENERATION (Megawatthours)": 100.0},
    ]
    pd.DataFrame(rows).to_excel(test_path, sheet_name="2002_Final", index=False)

    raw = load_raw_sheets(test_path, state="CA")
    grouped = pivot_by_source_group(raw)
    row = grouped.loc["2002-01-01"]

    assert row["Fossil"] == 100.0
    assert pd.isna(row["EIA_Reported_Total"])
    assert pd.isna(row["Excluded_MWh"])


def test_sheets_with_preamble_rows_are_not_silently_dropped():
    """Regression test for the actual production bug: sheets from 2012
    onward have 3 title/metadata rows (DOE header, description, source
    citation) above the real column header row, while 2001-2011 sheets
    have the header on row 0. Reading every sheet with a fixed header row
    silently dropped every post-2011 sheet, since they never had a real
    'ENERGY SOURCE' column at row 0.
    """
    test_dir = Path(tempfile.mkdtemp())
    test_path = test_dir / "gen_mixed_preamble.xlsx"

    old_style = pd.DataFrame([
        ["YEAR", "MONTH", "STATE", "TYPE OF PRODUCER", "ENERGY SOURCE", "GENERATION (Megawatthours)"],
        [2001, 1, "CA", "Total Electric Power Industry", "Coal", 199857],
        [2001, 1, "CA", "Total Electric Power Industry", "Total", 199857],
    ])

    new_style = pd.DataFrame([
        ["U.S. Department of Energy, The Energy Information Administration (EIA)", None, None, None, None, None],
        ["Monthly Generation Data by State, Producer Sector and Energy Source; Final 2012", None, None, None, None,
         None],
        ["Sources: EIA-923 Report", None, None, None, None, None],
        ["YEAR", "MONTH", "STATE", "TYPE OF PRODUCER", "ENERGY SOURCE", "GENERATION (Megawatthours)"],
        [2012, 1, "CA", "Total Electric Power Industry", "Coal", 200000],
        [2012, 1, "CA", "Total Electric Power Industry", "Total", 200000],
    ])

    with pd.ExcelWriter(test_path) as writer:
        old_style.to_excel(writer, sheet_name="2001_2002_FINAL", index=False, header=False)
        new_style.to_excel(writer, sheet_name="2012_Final", index=False, header=False)

    raw = load_raw_sheets(test_path, state="CA")
    years_found = sorted(raw["YEAR"].astype(int).unique())

    assert years_found == [2001, 2012], f"expected both years present, got {years_found}"

    grouped = pivot_by_source_group(raw)
    assert grouped.loc["2001-01-01", "Fossil"] == 199857
    assert grouped.loc["2012-01-01", "Fossil"] == 200000


def test_inconsistent_generation_column_whitespace_across_years_is_not_lost():
    """
    Regression test for the actual production bug: sheets from 2015
    onward use 'GENERATION  (Megawatthours)' (double space) while
    2001-2014 use 'GENERATION (Megawatthours)' (single space). Without
    normalizing whitespace, pd.concat() treats these as two different
    columns, silently zeroing out every 2015+ row when only the
    single-space column name is read downstream.
    """
    test_dir = Path(tempfile.mkdtemp())
    test_path = test_dir / "gen_whitespace.xlsx"

    old_style = pd.DataFrame([
        ["YEAR", "MONTH", "STATE", "TYPE OF PRODUCER", "ENERGY SOURCE", "GENERATION (Megawatthours)"],
        [2014, 12, "CA", "Total Electric Power Industry", "Coal", 500000],
        [2014, 12, "CA", "Total Electric Power Industry", "Total", 500000],
    ])
    new_style = pd.DataFrame([
        ["U.S. Department of Energy, The Energy Information Administration (EIA)", None, None, None, None, None],
        ["Monthly Generation Data by State, Producer Sector and Energy Source; Months Through December 2015", None,
         None, None, None, None],
        ["Sources: EIA-923 Report", None, None, None, None, None],
        ["YEAR", "MONTH", "STATE", "TYPE OF PRODUCER", "ENERGY SOURCE", "GENERATION  (Megawatthours)"],
        [2015, 1, "CA", "Total Electric Power Industry", "Coal", 600000],
        [2015, 1, "CA", "Total Electric Power Industry", "Total", 600000],
    ])

    with pd.ExcelWriter(test_path) as writer:
        old_style.to_excel(writer, sheet_name="2014_Final", index=False, header=False)
        new_style.to_excel(writer, sheet_name="2015_Final", index=False, header=False)

    raw = load_raw_sheets(test_path, state="CA")
    assert "GENERATION (Megawatthours)" in raw.columns
    assert "GENERATION  (Megawatthours)" not in raw.columns

    grouped = pivot_by_source_group(raw)
    assert grouped.loc["2014-12-01", "Fossil"] == 500000
    assert grouped.loc["2015-01-01", "Fossil"] == 600000
    assert grouped.loc["2015-01-01", "EIA_Reported_Total"] == 600000
