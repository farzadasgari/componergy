"""
Hovmoller diagrams of STI, SAPEI and SCDHI.

Four variants of one plot, matching the published captions:
    1. annual  x latitude
    2. annual  x longitude
    3. monthly x latitude
    4. monthly x longitude
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "dejavuserif",
    "savefig.dpi": 600,
})

import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def reduce_for_hovmoller(da: xr.DataArray, mode: str, space: str) -> xr.DataArray:
    """Average out one spatial axis, then aggregate time to annual or keep monthly."""
    if space == "lat":
        reduced = da.mean("lon")
    elif space == "lon":
        reduced = da.mean("lat")
    else:
        raise ValueError("space must be 'lat' or 'lon'")

    if mode == "monthly":
        return reduced.transpose("time", space)

    if mode == "annual":
        annual = reduced.groupby("time.year").mean("time")
        return annual.transpose("year", space)

    raise ValueError("mode must be 'annual' or 'monthly'")


def symmetric_levels(da: xr.DataArray, n_levels=21, percentile=98) -> np.ndarray:
    """Contour levels symmetric about zero, scaled to a high percentile of |value|."""
    vals = np.abs(da.values[np.isfinite(da.values)])
    if len(vals) == 0:
        return np.linspace(-1.0, 1.0, n_levels)
    vmax = float(np.percentile(vals, percentile))
    vmax = max(vmax, 1e-6)
    return np.linspace(-vmax, vmax, n_levels)
