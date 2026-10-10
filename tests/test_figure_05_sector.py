import numpy as np
import pandas as pd

from componergy.figures.sector import (
    aggregate_consumption_by_sector, prepare_sector_data, compute_sector_composites, SECTOR_ORDER,
)


def test_aggregate_consumption_by_sector_conserves_total():
    idx_rows = [
        {"Date": pd.Timestamp("2008-01-01"), "CountyName": "ALPHA", "Sector": "Residential",
         "Consumption (MWh)": 1000.0},
        {"Date": pd.Timestamp("2008-01-01"), "CountyName": "BETA", "Sector": "Residential", "Consumption (MWh)": 500.0},
        {"Date": pd.Timestamp("2008-01-01"), "CountyName": "ALPHA", "Sector": "Commercial",
         "Consumption (MWh)": 2000.0},
    ]
    long_df = pd.DataFrame(idx_rows)

    sector_gwh = aggregate_consumption_by_sector(long_df)

    raw_total_gwh = long_df["Consumption (MWh)"].sum() / 1000.0
    sector_total_gwh = sector_gwh.loc["2008-01-01"].sum()
    assert abs(raw_total_gwh - sector_total_gwh) < 1e-9


def test_sector_order_matches_real_tcu_naming_not_stale_alias():
    assert "TCU" in SECTOR_ORDER
    assert "Transportation, Communication, and Utilities" not in SECTOR_ORDER

    rng = np.random.default_rng(0)
    n = 30
    idx = pd.date_range("2008-01-01", periods=n, freq="MS")
    rows = []
    for sector in ["TCU", "Streetlighting", "Residential", "Mining", "Commercial", "Industrial", "Ag With Water Pump"]:
        for date in idx:
            rows.append({"Date": date, "CountyName": "ALPHA", "Sector": sector,
                         "Consumption (MWh)": float(rng.normal(1000, 50))})
    long_df = pd.DataFrame(rows)
    sector_gwh = aggregate_consumption_by_sector(long_df)

    ordered = [s for s in SECTOR_ORDER if s in sector_gwh.columns] + \
              [s for s in sector_gwh.columns if s not in SECTOR_ORDER]
    assert ordered.index("TCU") == 4


def test_compute_sector_composites_fdr_corrects_across_all_sector_event_tests():
    rng = np.random.default_rng(1)
    n = 200
    idx = pd.date_range("2008-01-01", periods=n, freq="MS")

    event_mask = pd.Series([True] * 20 + [False] * (n - 20), index=idx)
    normal_mask = pd.Series([False] * 20 + [True] * (n - 20), index=idx)

    raw_a = pd.Series(rng.normal(1000, 20, n), index=idx)
    raw_a.iloc[:20] += 200
    raw_b = pd.Series(rng.normal(500, 20, n), index=idx)

    sector_raw = pd.DataFrame({"SectorA": raw_a, "SectorB": raw_b})
    sector_z = sector_raw.apply(lambda s: (s - s.mean()) / s.std())

    data = {
        "sector_raw": sector_raw,
        "sector_z": sector_z,
        "event_masks": {"Heatwave": event_mask},
        "normal": normal_mask,
    }

    comp = compute_sector_composites(data)
    assert len(comp) == 2
    assert "q_value" in comp.columns

    row_a = comp[comp["Sector"] == "SectorA"].iloc[0]
    row_b = comp[comp["Sector"] == "SectorB"].iloc[0]
    assert row_a["q_value"] < 0.05
    assert row_b["p_value"] > row_a["p_value"]
    assert abs(row_b["Delta_GWh"]) < 0.2 * abs(row_a["Delta_GWh"])
    assert (comp["Delta_GWh"] == comp["Delta_native"]).all()
