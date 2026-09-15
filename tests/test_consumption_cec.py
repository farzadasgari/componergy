import pandas as pd
from pathlib import Path
import tempfile

from componergy.preprocessing.consumption_cec import load_raw_consumption


def _make_test_workbook():
    test_dir = Path(tempfile.mkdtemp())
    test_path = test_dir / "consumption.xlsx"

    rows = [
        ["Monthly Sector-level Electricity Consumption by County and Planning Area, 2008-2024",
         None, None, None, None, None, None, None],
        ["Year", "Month", "Planning_area", "CountyNum", "CountyName", "Sector", "Consumption (MWh)",
         "Consumption (KWh)"],
        [2008, 1, "BUGL", 19, "LOS ANGELES", "Residential", 1000.0, 1000000.0],
        [2008, 1, "SCE", 19, "LOS ANGELES", "Residential", 500.0, 500000.0],
        [2008, 1, "BUGL", 19, "LOS ANGELES", "Commercial", 2000.0, 2000000.0],
        [2008, 1, "XYZ", 1, "ALAMEDA", "Residential", 300.0, 300000.0],
    ]
    pd.DataFrame(rows).to_excel(test_path, sheet_name="Electricity Consumption", index=False, header=False)
    return test_path
