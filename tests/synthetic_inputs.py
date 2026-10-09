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
