"""Synthetic inputs shaped like the real pipeline's, for end-to-end tests of the figure scripts."""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon

REGION_NAMES = ["North Coast", "Sacramento-Delta", "Central Coast", "San Joaquin Valley", "South Coast"]
SECTORS = ["Residential", "Commercial", "Industrial", "Ag With Water Pump", "TCU", "Mining", "Streetlighting"]
SOURCE_COLUMNS = ["Fossil", "Hydro", "Solar", "Wind", "Nuclear", "Other Renewable"]


def _ar1(n, phi, rng):
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + rng.normal()
    return e


def make_indices(start="1951-01-01", end="2025-08-01", seed=1, n_lat=6, n_lon=6, warming_per_decade=0.16):
    """STI (warming trend), SAPEI and SCDHI on a small lat/lon grid with a shared statewide signal."""
    times = pd.date_range(start, end, freq="MS")
    n = len(times)
    rng = np.random.default_rng(seed)
    yr = np.asarray(times.year + (times.month - 0.5) / 12.0, dtype=float)
    trend = warming_per_decade * (yr - yr.mean()) / 10.0
    heat_common = _ar1(n, 0.5, rng)[:, None, None]
    dry_common = (-0.4 * heat_common[:, 0, 0] + _ar1(n, 0.5, rng))[:, None, None]
    shape = (n, n_lat, n_lon)
    sti = trend[:, None, None] + heat_common + 0.4 * rng.normal(size=shape)
    sapei = dry_common + 0.4 * rng.normal(size=shape)
    scdhi = 0.6 * sapei - 0.4 * sti + 0.3 * rng.normal(size=shape)
    coords = {"time": times, "lat": np.linspace(32.5, 42.0, n_lat), "lon": np.linspace(-124.0, -114.5, n_lon)}
    return xr.Dataset({"sti": (("time", "lat", "lon"), sti), "sapei_3m": (("time", "lat", "lon"), sapei),
                       "scdhi_3m": (("time", "lat", "lon"), scdhi)}, coords=coords)


def make_generation(start="2001-01-01", end="2025-08-01", seed=2):
    idx = pd.date_range(start, end, freq="MS")
    n = len(idx)
    rng = np.random.default_rng(seed)
    season = np.sin(2 * np.pi * (idx.month.values - 4) / 12)
    t = np.linspace(0, 1, n)
    shares = {"Fossil": 0.55 + 0.05 * season, "Hydro": 0.15 - 0.05 * season, "Solar": 0.01 + 0.12 * t,
              "Wind": 0.04 + 0.03 * t, "Nuclear": np.full(n, 0.08), "Other Renewable": np.full(n, 0.07)}
    total = 19e6 * (1 + 0.1 * season + rng.normal(0, 0.03, n))
    df = pd.DataFrame({k: v * total * (1 + rng.normal(0, 0.02, n)) for k, v in shares.items()}, index=idx)
    df["Total"] = df[SOURCE_COLUMNS].sum(axis=1)
    return df


def make_demand_mwh(generation_df, import_fraction=0.28, seed=3, start=None, end=None, noise=0.02):
    rng = np.random.default_rng(seed)
    d = generation_df["Total"] / (1.0 - import_fraction) * (1 + rng.normal(0, noise, len(generation_df)))
    return d.loc[start:end] if start or end else d


def make_consumption_long(n_counties=10, start="2008-01-01", end="2024-12-01", seed=4):
    dates = pd.date_range(start, end, freq="MS")
    n = len(dates)
    rng = np.random.default_rng(seed)
    season = np.sin(2 * np.pi * (dates.month.values - 4) / 12)
    rows = []
    for c in range(n_counties):
        for s_i, sector in enumerate(SECTORS):
            base = 5000.0 * (1 + 0.3 * s_i) * (1 + c / 10)
            vals = base * (1 + (0.15 if s_i < 2 else 0.05) * season + rng.normal(0, 0.03, n))
            rows.append(pd.DataFrame({"Date": dates, "CountyName": f"COUNTY{c:02d}", "Sector": sector,
                                      "Consumption (MWh)": vals}))
    return pd.concat(rows, ignore_index=True)
