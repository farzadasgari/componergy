import numpy as np
import pandas as pd
import xarray as xr

from componergy.figures.sites import extract_site_series, REPRESENTATIVE_SITES


def _ds(n_months=60):
    times = pd.date_range("2008-01-01", periods=n_months, freq="MS")
    lats = np.linspace(32.5, 42.0, 20)
    lons = np.linspace(-124.5, -114.0, 20)
    rng = np.random.default_rng(0)
    shape = (n_months, 20, 20)
    return xr.Dataset(
        {"sti": (("time", "lat", "lon"), rng.normal(0, 1, shape)),
         "sapei_3m": (("time", "lat", "lon"), rng.normal(0, 1, shape)),
         "scdhi_3m": (("time", "lat", "lon"), rng.normal(0, 1, shape))},
        coords={"time": times, "lat": lats, "lon": lons},
    )


def test_extract_site_series_returns_all_sites_and_indices():
    ds = _ds()
    df = extract_site_series(ds, {"STI": "sti", "SAPEI": "sapei_3m", "SCDHI": "scdhi_3m"})

    assert set(df["Site"].unique()) == {s[0] for s in REPRESENTATIVE_SITES}
    for col in ("STI", "SAPEI", "SCDHI", "Date", "Grid_Lat", "Grid_Lon"):
        assert col in df.columns
    assert len(df) == len(REPRESENTATIVE_SITES) * 60


def test_extract_site_series_records_grid_offset_for_auditability():
    ds = _ds()
    df = extract_site_series(ds, {"STI": "sti"})
    meta = df.groupby("Site")[["Requested_Lat", "Grid_Lat", "Requested_Lon", "Grid_Lon"]].first()
    assert (np.abs(meta["Grid_Lat"] - meta["Requested_Lat"]) < 0.5).all()
    assert (np.abs(meta["Grid_Lon"] - meta["Requested_Lon"]) < 0.5).all()


def test_extract_site_series_picks_distinct_cells_for_distinct_sites():
    ds = _ds()
    df = extract_site_series(ds, {"STI": "sti"})
    cells = df.groupby("Site")[["Grid_Lat", "Grid_Lon"]].first()
    assert len(cells.drop_duplicates()) == len(REPRESENTATIVE_SITES)
