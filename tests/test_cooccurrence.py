import numpy as np
import pandas as pd

from componergy.figures.cooccurrence import (
    aggregate_consumption_to_region_sector, compute_cooccurrence_frequency, S1_SECTORS,
)


def _consumption(n_months=120):
    dates = pd.date_range("2008-01-01", periods=n_months, freq="MS")
    rng = np.random.default_rng(0)
    rows = []
    for date in dates:
        for county, region in [("ALPHA", "R1"), ("BETA", "R2")]:
            for sector in ["Residential", "Commercial", "Industrial", "Ag With Water Pump"]:
                rows.append({"Date": date, "CountyName": county, "Sector": sector,
                             "Consumption (MWh)": float(rng.normal(1000, 100))})
    return pd.DataFrame(rows), pd.DataFrame({"CountyName": ["ALPHA", "BETA"], "Region": ["R1", "R2"]})


def test_region_sector_aggregation_preserves_total_and_splits_sectors():
    long_df, mapping = _consumption(24)
    out = aggregate_consumption_to_region_sector(long_df, mapping)

    assert out.columns.nlevels == 2
    assert ("R1", "Residential") in out.columns
    jan_total_gwh = long_df[long_df["Date"] == "2008-01-01"]["Consumption (MWh)"].sum() / 1000.0
    assert abs(out.loc["2008-01-01"].sum() - jan_total_gwh) < 1e-9


def test_region_sector_aggregation_raises_on_unmapped_county():
    long_df, mapping = _consumption(12)
    bad_map = mapping[mapping["CountyName"] != "BETA"]
    try:
        aggregate_consumption_to_region_sector(long_df, bad_map)
        assert False, "should have raised for the unmapped county"
    except ValueError as e:
        assert "BETA" in str(e)


def test_cooccurrence_frequency_is_high_when_demand_spikes_during_cdhe():
    n = 180
    dates = pd.date_range("2008-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)

    scdhi = pd.Series(rng.normal(0, 1, n), index=dates)
    cdhe_months = np.zeros(n, dtype=bool)
    cdhe_months[:20] = True
    scdhi[cdhe_months] -= 4.0
    regional_scdhi = pd.DataFrame({"R1": scdhi})

    resid = pd.Series(rng.normal(1000, 20, n), index=dates)
    resid[cdhe_months] += 400.0
    comm = pd.Series(rng.normal(1000, 20, n), index=dates)

    cols = pd.MultiIndex.from_tuples([("R1", "Residential"), ("R1", "Commercial")])
    region_sector = pd.DataFrame(np.column_stack([resid.values, comm.values]), index=dates, columns=cols)

    out = compute_cooccurrence_frequency(regional_scdhi, region_sector,
                                         sectors={"Residential": "Residential", "Commercial": "Commercial"})
    freq = out.set_index("Sector")["Cooccurrence_Frequency"]
    assert freq["Residential"] > 0.5
    assert freq["Commercial"] < freq["Residential"]


def test_cooccurrence_reports_nan_for_missing_region_sector_pair():
    n = 60
    dates = pd.date_range("2008-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    regional_scdhi = pd.DataFrame({"R1": pd.Series(rng.normal(0, 1, n), index=dates)})
    cols = pd.MultiIndex.from_tuples([("R1", "Residential")])
    region_sector = pd.DataFrame(rng.normal(1000, 20, (n, 1)), index=dates, columns=cols)

    out = compute_cooccurrence_frequency(regional_scdhi, region_sector, sectors=S1_SECTORS)
    missing = out[out["Sector"] == "Industrial"]["Cooccurrence_Frequency"]
    assert missing.isna().all()
