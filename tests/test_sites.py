import numpy as np
import pytest
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


def test_extract_site_series_moves_off_an_empty_cell_to_the_nearest_cell_with_data():
    ds = _ds()
    name, lat, lon = REPRESENTATIVE_SITES[0]
    i = int(np.abs(ds["lat"].values - lat).argmin())
    j = int(np.abs(ds["lon"].values - lon).argmin())
    for var in ds.data_vars:
        ds[var][:, i, j] = np.nan
    df = extract_site_series(ds, {"STI": "sti", "SAPEI": "sapei_3m", "SCDHI": "scdhi_3m"},
                             sites=[(name, lat, lon)])
    assert (df["Grid_Lat"].iloc[0], df["Grid_Lon"].iloc[0]) != (ds["lat"].values[i], ds["lon"].values[j])
    assert df[["STI", "SAPEI", "SCDHI"]].notna().any().all()


def test_no_representative_site_samples_an_empty_cell():
    from componergy.paths import INDICES_FILE
    if not INDICES_FILE.exists():
        pytest.skip(f"{INDICES_FILE} not on disk")
    with xr.open_dataset(INDICES_FILE) as ds:
        df = extract_site_series(ds, {"STI": "sti", "SAPEI": "sapei_3m", "SCDHI": "scdhi_3m"})
    empty = [site for site, sub in df.groupby("Site")
             if not sub[["STI", "SAPEI", "SCDHI"]].notna().any().all()]
    assert not empty, f"sites sampling an empty grid cell: {empty}"
