import pandas as pd
from pathlib import Path
import tempfile

from componergy.preprocessing.demand_eia861m import load_raw_demand_sheet, combine_demand_eras


def _make_test_workbook_with_embedded_headers():
    test_dir = Path(tempfile.mkdtemp())
    test_path = test_dir / "demand_test.xlsx"

    header_sector = ["", "", "", "", "RESIDENTIAL", "", "", "", "TOTAL", "", "", ""]
    header_subcol = ["", "", "", "", "Revenue", "Sales", "Customers", "Price", "Revenue", "Sales", "Customers", "Price"]
    header_labels = ["Year", "Month", "State", "Data Status", "Thousand Dollars", "Megawatthours", "Count", "Cents/kWh",
                     "Thousand Dollars", "Megawatthours", "Count", "Cents/kWh"]

    rows = [header_sector, header_subcol, header_labels]
    rows.append([2020, 1, "CA", "Final", 100, 200, 300, 10.0, 500, 20000000, 3000, 15.0])
    rows.append([2020, 2, "CA", "Final", 110, 210, 310, 10.1, 510, 21000000, 3010, 15.1])

    header_sector_shifted = ["", "", "", "RESIDENTIAL", "", "", "", "TOTAL", "", "", "", ""]
    header_subcol_shifted = ["", "", "", "Revenue", "Sales", "Customers", "Price", "Revenue", "Sales", "Customers",
                             "Price", ""]
    header_labels_shifted = ["Year", "Month", "State", "Data Status", "Thousand Dollars", "Megawatthours", "Count",
                             "Cents/kWh", "Thousand Dollars", "Megawatthours", "Count", "Cents/kWh"]
    rows.append(header_sector_shifted)
    rows.append(header_subcol_shifted)
    rows.append(header_labels_shifted)

    rows.append([2020, 3, "CA", "Final", 120, 220, 320, 10.2, 520, 22000000, 3020, 15.2])

    pd.DataFrame(rows).to_excel(test_path, sheet_name="Monthly-States", index=False, header=False)
    return test_path


def test_embedded_header_with_different_offset_is_correctly_skipped():
    path = _make_test_workbook_with_embedded_headers()
    result = load_raw_demand_sheet(path, "Monthly-States")

    assert len(result) == 3
    assert result.loc["2020-01-01"] == 20000000
    assert result.loc["2020-02-01"] == 21000000
    assert result.loc["2020-03-01"] == 22000000


def test_state_filtering_excludes_other_states():
    test_dir = Path(tempfile.mkdtemp())
    test_path = test_dir / "demand_multi_state.xlsx"

    rows = [
        ["", "", "", "", "RESIDENTIAL", "", "", "", "TOTAL", "", "", ""],
        ["", "", "", "", "Revenue", "Sales", "Customers", "Price", "Revenue", "Sales", "Customers", "Price"],
        ["Year", "Month", "State", "Data Status", "Thousand Dollars", "Megawatthours", "Count", "Cents/kWh",
         "Thousand Dollars", "Megawatthours", "Count", "Cents/kWh"],
        [2020, 1, "CA", "Final", 100, 200, 300, 10.0, 500, 20000000, 3000, 15.0],
        [2020, 1, "TX", "Final", 999, 999, 999, 99.0, 999, 99999999, 999, 99.0],
    ]
    pd.DataFrame(rows).to_excel(test_path, sheet_name="Monthly-States", index=False, header=False)

    result = load_raw_demand_sheet(test_path, "Monthly-States", state="CA")
    assert len(result) == 1
    assert result.iloc[0] == 20000000


def test_combine_demand_eras_prefers_newer_file_on_overlap():
    old_series = pd.Series(
        [100.0, 200.0, 300.0],
        index=pd.to_datetime(["2009-10-01", "2009-11-01", "2009-12-01"]),
    )
    new_series = pd.Series(
        [999.0, 400.0],
        index=pd.to_datetime(["2009-12-01", "2010-01-01"]),
    )

    combined = combine_demand_eras(old_series, new_series)

    assert len(combined) == 4
    assert combined.loc["2009-12-01"] == 999.0
    assert combined.loc["2009-10-01"] == 100.0
    assert combined.loc["2010-01-01"] == 400.0
