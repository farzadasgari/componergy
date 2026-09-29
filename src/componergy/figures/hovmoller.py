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


def plot_hovmoller(fields, mode: str, space: str, figsize=(18, 20), xtick_rotation=90):
    """Stacked Hovmoller panels, one per (title, DataArray, cmap) entry in `fields`."""
    fig, axes = plt.subplots(nrows=len(fields), ncols=1, figsize=figsize,
                             sharex=True, constrained_layout=True)
    if len(fields) == 1:
        axes = [axes]

    for ax, (title, da, cmap) in zip(axes, fields):
        reduced = reduce_for_hovmoller(da, mode=mode, space=space)
        levels = symmetric_levels(reduced)
        y = reduced[space].values

        if mode == "annual":
            x = reduced["year"].values
            X, Y = np.meshgrid(x, y)
            Z = reduced.transpose(space, "year").values
        else:
            x = mdates.date2num(reduced["time"].values)
            X, Y = np.meshgrid(x, y)
            Z = reduced.transpose(space, "time").values

        im = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend="both")

        if mode == "monthly":
            ax.xaxis_date()
            ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax.set_ylabel("Latitude (°N)" if space == "lat" else "Longitude (°E)", fontsize=24)
        ax.tick_params(labelsize=22, width=1.5, length=6)

        cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.01, shrink=0.95)
        cbar.set_label(title, fontsize=28)
        cbar.ax.tick_params(labelsize=26, width=1.5, length=6)

    axes[-1].set_xlabel("Time (year)", fontsize=28)
    axes[-1].tick_params(axis="x", labelrotation=xtick_rotation)
    for lbl in axes[-1].get_xticklabels():
        lbl.set_ha("right")

    return fig


HOVMOLLER_VARIANTS = [
    ("S2", "annual", "lat", "Latitudinal evolution of climate extremes (annual)"),
    ("S3", "annual", "lon", "Longitudinal evolution of climate extremes (annual)"),
    ("S4", "monthly", "lat", "Seasonal and latitudinal variability of climate extremes"),
    ("S5", "monthly", "lon", "Seasonal and longitudinal variability of climate extremes"),
]
