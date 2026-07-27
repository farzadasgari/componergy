import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

plt.rcParams["font.family"] = "Cambria"

def _to_monthly_time(da: xr.DataArray) -> xr.DataArray:
    if "time" in da.dims:
        return da
    if {"year", "month"}.issubset(da.dims):
        s = da.stack(time=("year", "month"))
        s = s.reset_index("time")
        t = pd.to_datetime({"year": s["year"].values, "month": s["month"].values, "day": 1})
        s = s.assign_coords(time=("time", t)).drop_vars(["year", "month"], errors="ignore")
        return s.sortby("time")
    raise ValueError(f"Unrecognized time layout. dims={da.dims}")

def reduce_for_hovmoller(da: xr.DataArray, mode: str, space: str) -> xr.DataArray:
    if space == "lat":
        r = da.mean("lon")
    elif space == "lon":
        r = da.mean("lat")
    else:
        raise ValueError("space must be 'lat' or 'lon'")

    if mode == "monthly":
        r = _to_monthly_time(r)
        return r.transpose("time", space)

    if mode == "annual":
        if "time" in r.dims:
            a = r.groupby("time.year").mean("time")
            return a.transpose("year", space)
        if {"year", "month"}.issubset(r.dims):
            a = r.mean("month")
            return a.transpose("year", space)
        raise ValueError(f"Unrecognized time layout. dims={r.dims}")

    raise ValueError("mode must be 'annual' or 'monthly'")

def plot_hovmoller(fields, mode: str, space: str, figsize=(18, 20), out=None, dpi=300, xtick_rotation=90):
    fig, axes = plt.subplots(nrows=len(fields), ncols=1, figsize=figsize, sharex=True, constrained_layout=True)
    if len(fields) == 1:
        axes = [axes]

    for ax, (title, da, levels, cmap) in zip(axes, fields):
        out_da = reduce_for_hovmoller(da, mode=mode, space=space)
        y = out_da[space].values

        if mode == "annual":
            x = out_da["year"].values
            X, Y = np.meshgrid(x, y)
            Z = out_da.transpose(space, "year").values
            im = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend="both")
        else:
            times = out_da["time"].values
            x = mdates.date2num(times)
            X, Y = np.meshgrid(x, y)
            Z = out_da.transpose(space, "time").values
            im = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend="both")
            ax.xaxis_date()
            ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax.set_ylabel("Latitude (°N)" if space == "lat" else "Longitude (°E)", fontsize=24)
        ax.tick_params(labelsize=22, width=1.5, length=6)

        cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.01, shrink=0.95)
        cbar.set_label(title, fontsize=28)
        cbar.ax.tick_params(labelsize=26, width=1.5, length=6)

    axes[-1].set_xlabel("Time (year)" if mode == "annual" else "Time (month)", fontsize=28)
    axes[-1].tick_params(axis="x", labelrotation=xtick_rotation)
    for lbl in axes[-1].get_xticklabels():
        lbl.set_ha("right")

    if out is not None:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axes

ds = xr.open_dataset("../dataset/california_monthly.nc")
fields = [
    ("SCDHI", ds["scdhi"], np.linspace(-1.0, 1.0, 21), "RdBu"),
    ("SCEI",  ds["scei"],  np.linspace(-1.0, 1.0, 21), "RdBu"),
    ("SDHI",  ds["sdhi1"], np.linspace(-1.0, 1.0, 21), "RdBu"),
]

plot_hovmoller(fields, mode="monthly", space="lat", figsize=(18, 20), out="../plots/hovmoller_monthly_lat.png")
plot_hovmoller(fields, mode="monthly", space="lon", figsize=(18, 20), out="../plots/hovmoller_monthly_lon.png")

plot_hovmoller(fields, mode="annual", space="lat", figsize=(18, 20), out="../plots/hovmoller_annual_lat.png")
plot_hovmoller(fields, mode="annual", space="lon", figsize=(18, 20), out="../plots/hovmoller_annual_lon.png")
