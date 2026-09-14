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
