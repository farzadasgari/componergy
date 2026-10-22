import numpy as np
import pandas as pd

from tests.synthetic_inputs import make_no_event_effect_series
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
    raw_b = make_no_event_effect_series(rng.normal(500, 20, n), idx, event_mask)

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

    assert np.all(comp["q_value"].values >= comp["p_value"].values - 1e-12)

    row_a = comp[comp["Sector"] == "SectorA"].iloc[0]
    row_b = comp[comp["Sector"] == "SectorB"].iloc[0]
    assert abs(row_a["q_value"] - len(comp) * row_a["p_value"]) < 1e-12
    assert row_a["q_value"] < 0.05
    assert row_b["q_value"] > 0.05
    assert abs(row_b["Delta_GWh"]) < 0.2 * abs(row_a["Delta_GWh"])
    assert (comp["Delta_GWh"] == comp["Delta_native"]).all()


def test_prepare_sector_data_heatwave_and_drought_are_mutually_exclusive():
    import xarray as xr

    rng = np.random.default_rng(2)
    n = 120
    times = pd.date_range("2008-01-01", periods=n, freq="MS")
    lats = np.linspace(33, 41, 5)
    lons = np.linspace(-123, -115, 5)

    indices_ds = xr.Dataset(
        {"sti": (("time", "lat", "lon"), rng.normal(0, 1, (n, 5, 5))),
         "sapei_3m": (("time", "lat", "lon"), rng.normal(0, 1, (n, 5, 5))),
         "scdhi_3m": (("time", "lat", "lon"), rng.normal(0, 1, (n, 5, 5)))},
        coords={"time": times, "lat": lats, "lon": lons},
    )

    rows = []
    for date in times:
        for sector in ["Residential", "Commercial"]:
            rows.append({"Date": date, "CountyName": "ALPHA", "Sector": sector,
                         "Consumption (MWh)": float(rng.normal(1000, 50))})
    long_df = pd.DataFrame(rows)

    data = prepare_sector_data(indices_ds, long_df)
    overlap = (data["event_masks"]["Heatwave"] & data["event_masks"]["Drought"]).sum()
    assert overlap == 0


def _wheel_frame():
    rows = []
    for ev, base in (("Heatwave", 1.0), ("Drought", -1.0), ("Compound", 2.0)):
        for k, sec in enumerate(["Residential", "Commercial", "Industrial"]):
            rows.append({"Event": ev, "Sector": sec, "Delta_GWh": base * (k + 1) * 10.0, "Delta_z": base * 0.2 * (k + 1),
                         "q_value": 0.01 if (ev, k) == ("Compound", 2) else 0.5,
                         "p_exact": 0.03 if (ev, k) == ("Drought", 0) else 0.4})
    return pd.DataFrame(rows)


def test_draw_sector_wheel_wedges_follow_table():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from componergy.figures.sector import draw_sector_wheel, wedge_thickness, RING_BASE

    comp = _wheel_frame()
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    sc = draw_sector_wheel(ax, comp, ["Residential", "Commercial", "Industrial"])
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rects) == 9
    assert min(p.get_y() for p in rects) == RING_BASE
    heights = sorted(p.get_height() for p in rects)
    expected = sorted(wedge_thickness(g, sc["gmax"]) for g in comp["Delta_GWh"])
    assert np.allclose(heights, expected, rtol=1e-12, atol=0)
    solid = [p for p in rects if p.get_linewidth() > 0 and p.get_linestyle() == "-"]
    dashed = [p for p in rects if p.get_linewidth() > 0 and p.get_linestyle() == "--"]
    assert len(solid) == 1 and len(dashed) == 1
    assert sum(p.get_linewidth() == 0 for p in rects) == 7
    plt.close(fig)


def test_wedge_outline_ignores_marginal_q():
    from componergy.figures.sector import wedge_outline
    assert wedge_outline(0.04, 0.001)[1] == "-"
    assert wedge_outline(0.07, 0.04)[1] == "--"
    assert wedge_outline(0.07, 0.06)[2] == 0.0
    assert wedge_outline(0.08, float("nan"))[2] == 0.0
