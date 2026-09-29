"""climate-extreme indices at six representative California locations."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Six locations spanning California's distinct climate zones.
# Coordinates are approximate city/area centroids.
REPRESENTATIVE_SITES = [
    ("Eureka (North Coast)", 40.80, -124.16),
    ("Sacramento (Sacramento Valley)", 38.58, -121.49),
    ("Fresno (San Joaquin Valley)", 36.75, -119.77),
    ("San Luis Obispo (Central Coast)", 35.28, -120.66),
    ("Los Angeles (South Coast)", 34.05, -118.24),
    ("Barstow (Mojave Desert)", 34.90, -117.02),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "dejavuserif",
    "savefig.dpi": 600,
})

import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def extract_site_series(ds: xr.Dataset, var_names, sites=None) -> pd.DataFrame:
    """Long-format monthly series for each site, one column per named index."""
    if sites is None:
        sites = REPRESENTATIVE_SITES

    frames = []
    for name, lat, lon in sites:
        point = ds.sel(lat=lat, lon=lon, method="nearest")
        actual_lat = float(point["lat"].values)
        actual_lon = float(point["lon"].values)

        df = pd.DataFrame({label: point[var].to_series() for label, var in var_names.items()})
        df["Site"] = name
        df["Requested_Lat"] = lat
        df["Requested_Lon"] = lon
        df["Grid_Lat"] = actual_lat
        df["Grid_Lon"] = actual_lon
        frames.append(df.reset_index().rename(columns={"index": "Date", "time": "Date"}))

    return pd.concat(frames, ignore_index=True)


def plot_site_timeseries(site_df: pd.DataFrame, sites=None, figsize=(16, 18)):
    """Figure S6: stacked per-site monthly time series of STI, SAPEI and SCDHI."""
    if sites is None:
        sites = REPRESENTATIVE_SITES
    names = [s[0] for s in sites]

    colors = {"STI": "#c1272d", "SAPEI": "#0b6fa4", "SCDHI": "#6a3d9a"}
    fig, axes = plt.subplots(nrows=len(names), ncols=1, figsize=figsize,
                             sharex=True, constrained_layout=True)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        sub = site_df[site_df["Site"] == name].sort_values("Date")
        for label in ("STI", "SAPEI", "SCDHI"):
            if label in sub.columns:
                ax.plot(sub["Date"], sub[label], lw=1.5, alpha=0.9,
                        color=colors[label], label=label)
        ax.axhline(0, color="k", lw=0.9, alpha=0.35, zorder=0)
        ax.set_ylabel("Index value", fontsize=14)
        ax.set_title(name, fontsize=15, pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=12)

    axes[0].legend(frameon=False, ncol=3, fontsize=13, loc="upper left")
    axes[-1].set_xlabel("Time (year)", fontsize=15)
    return fig


def plot_site_scatter(site_df: pd.DataFrame, sites=None, figsize=(16, 11)):
    """Figure S7: per-site STI vs SAPEI scatter, SCDHI shown by colour and marker size."""
    if sites is None:
        sites = REPRESENTATIVE_SITES
    names = [s[0] for s in sites]

    scdhi_all = site_df["SCDHI"].values
    finite = scdhi_all[np.isfinite(scdhi_all)]
    vmax = float(np.percentile(np.abs(finite), 98)) if len(finite) else 1.0
    vmax = max(vmax, 1e-6)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu")

    n_cols = 3
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        sub = site_df[site_df["Site"] == name]
        sti, sapei, scdhi = sub["STI"].values, sub["SAPEI"].values, sub["SCDHI"].values
        sizes = 18 + 160 * (np.abs(scdhi) / vmax).clip(0, 1) ** 1.3

        ax.scatter(sti, sapei, c=scdhi, cmap=cmap, norm=norm, s=sizes,
                   alpha=0.72, edgecolors="none")
        ax.axvline(0, color="k", lw=0.9, alpha=0.35, zorder=0)
        ax.axhline(0, color="k", lw=0.9, alpha=0.35, zorder=0)
        ax.set_title(name, fontsize=14, pad=4)
        ax.set_xlabel("STI (heat)", fontsize=13)
        ax.set_ylabel("SAPEI (drought)", fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=11)

    for ax in axes[len(names):]:
        ax.set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation="vertical", shrink=0.85, pad=0.02)
    cbar.set_label("SCDHI (compound intensity)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    return fig
