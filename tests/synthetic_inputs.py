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
