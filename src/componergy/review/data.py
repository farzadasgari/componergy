# src/componergy/review/data.py
"""Inputs and outcome series for the review analyses, built with the figures' own prepare_* functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from componergy.figures.composite_stats import zscore

HEADLINE = ("Hydro share", "Fossil share", "Hydro GWh", "Fossil GWh", "Sonoran Desert", "Mojave Desert", "Imbalance")
GROUP_FAMILY = {"G-share": "generation", "G-energy": "generation", "D": "demand", "I": "imbalance"}
GROUP_TREND = {"G-share": "quadratic", "G-energy": "quadratic", "D": "linear", "I": "linear"}
GROUP_UNIT = {"G-share": "pp", "G-energy": "GWh", "D": "GWh", "I": "GWh"}
SONGS_STEP_START = pd.Timestamp("2012-02-01")
SOURCE_COLUMNS = ["Fossil", "Hydro", "Solar", "Wind", "Nuclear", "Other Renewable"]


def statewide_drivers(indices_ds, cache_path=None) -> pd.DataFrame:
    """
    Full-record z-scored statewide series exactly as figures.generation.prepare_data forms them
    (sti, sapei3, scdhi3, plus the new sapei12), and the raw statewide precipitation for the SPI check.
    """
    import os

    from componergy.figures.trend_stats import compute_area_weights, compute_statewide_mean

    from componergy.paths import INDICES_FILE

    if cache_path is not None and os.path.exists(cache_path) \
            and os.path.getmtime(cache_path) > os.path.getmtime(INDICES_FILE):
        return pd.read_csv(cache_path, parse_dates=[0], index_col=0)
    w = compute_area_weights(indices_ds["lat"])
    sw = lambda v: compute_statewide_mean(indices_ds[v], w).to_series()  # noqa: E731
    out = pd.DataFrame({"sti": zscore(sw("sti")), "sapei3": zscore(sw("sapei_3m")), "scdhi3": zscore(sw("scdhi_3m")),
                        "sapei12": zscore(sw("sapei_12m"))})
    if "prcp" in indices_ds:
        out["prcp_raw"] = sw("prcp")
    out.index = pd.DatetimeIndex(out.index.values, name=None)
    if cache_path is not None:
        from componergy.netcdf_io import atomic_to_csv
        atomic_to_csv(out, cache_path)
    return out


def smoothed_regressors(drivers: pd.DataFrame, index, smooth: int = 3) -> dict:
    """h = STI_sm, d = -SAPEI_sm (positive = drier), hd, d12 = -SAPEI12_sm, c = -SCDHI_sm on `index`."""
    sm = drivers[["sti", "sapei3", "scdhi3", "sapei12"]].rolling(smooth, center=True, min_periods=1).mean()
    sm = sm.reindex(pd.DatetimeIndex(index))
    h, d = sm["sti"], -sm["sapei3"]
    return {"h": h, "d": d, "hd": h * d, "d12": -sm["sapei12"], "c": -sm["scdhi3"]}


def songs_step(index) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    return pd.Series((idx >= SONGS_STEP_START).astype(float), index=idx)


def build_context(indices_ds, generation_df: pd.DataFrame, cec_demand: pd.Series, regions_gdf, county_region_map,
                  consumption_long_df, drivers: pd.DataFrame) -> dict:
    """Outcome series, event masks and regressors for the generation, demand and imbalance families."""
    from componergy.figures.generation import prepare_data as prepare_generation
    from componergy.figures.imbalance import prepare_imbalance_data
    from componergy.figures.zonal_consumption import prepare_zonal_data

    gen = prepare_generation(indices_ds, generation_df)
    zon = prepare_zonal_data(indices_ds, consumption_long_df, regions_gdf, county_region_map)
    imb = prepare_imbalance_data(indices_ds, generation_df, cec_demand.to_frame("total_sales"))

    gwh = (generation_df[SOURCE_COLUMNS] / 1000.0).rolling(3, center=True, min_periods=1).mean().loc[gen["mix_pct_sm"].index]
    energy = pd.DataFrame({"Fossil": gwh["Fossil"], "Hydro": gwh["Hydro"], "Nuclear": gwh["Nuclear"],
                           "Solar+Wind": gwh["Solar"] + gwh["Wind"], "Total": gwh.sum(axis=1)})

    gidx, didx = gen["mix_pct_sm"].index, zon["electricity_raw"].index
    assert imb["imb_gwh"].index.equals(didx), "imbalance and regional-demand windows differ"
    windows = {
        "generation": {"index": gidx, "masks": gen["event_masks"], "normal": gen["normal"], "meta": gen["event_meta"],
                       "flags": gen["event_flags"], "reg": smoothed_regressors(drivers, gidx)},
        "demand": {"index": didx, "masks": zon["event_masks"], "normal": zon["normal"], "meta": zon["event_meta"],
                   "flags": zon["event_flags"], "reg": smoothed_regressors(drivers, didx)},
        "imbalance": {"index": imb["imb_gwh"].index, "masks": imb["event_masks"], "normal": imb["normal"],
                      "meta": imb["event_meta"], "flags": imb["event_flags"],
                      "reg": smoothed_regressors(drivers, imb["imb_gwh"].index)},
    }
    outcomes = []
    for c in SOURCE_COLUMNS:
        outcomes.append(dict(name=f"{c} share", col=c, group="G-share", window="generation", y=gen["mix_pct_sm"][c]))
    for c in energy.columns:
        outcomes.append(dict(name=f"{c} GWh", col=c, group="G-energy", window="generation", y=energy[c]))
    for c in zon["electricity_raw"].columns:
        outcomes.append(dict(name=c, col=c, group="D", window="demand", y=zon["electricity_raw"][c]))
    outcomes.append(dict(name="Imbalance", col="Imbalance", group="I", window="imbalance", y=imb["imb_gwh"]))
    for o in outcomes:
        o.update(family=GROUP_FAMILY[o["group"]], trend=GROUP_TREND[o["group"]], unit=GROUP_UNIT[o["group"]])
    extras = {"nuclear_gwh": energy["Nuclear"], "songs": songs_step(gidx)}
    return {"outcomes": outcomes, "windows": windows, "energy": energy, "extras": extras,
            "raw_frames": {"G-share": gen["mix_pct_sm"], "G-energy": energy, "D": zon["electricity_raw"],
                           "I": imb["imb_gwh"].to_frame("Imbalance")}}


def load_all(drivers_cache=None) -> dict:
    """Open every input from `paths` and build the context (slow: opens the 6 GB indices file)."""
    import xarray as xr

    from componergy import paths
    from componergy.analysis.regions import build_county_to_region_map
    from componergy.download.boundaries import get_california_counties
    from componergy.download.climate_regions import get_california_climate_regions
    from componergy.figures.imbalance import load_demand_sources
    from componergy.preprocessing.consumption_cec import load_raw_consumption

    ds = xr.open_dataset(paths.INDICES_FILE)
    gen = pd.read_csv(paths.GENERATION_MONTHLY_FILE, parse_dates=[0], index_col=0)
    gen.index = pd.DatetimeIndex(gen.index.values, name=None)
    cec = load_demand_sources()["cec"]
    if cec is None:
        raise FileNotFoundError(f"the CEC county consumption workbook is required: {paths.COUNTY_CONSUMPTION_XLSX}")
    regions = get_california_climate_regions()
    cmap = build_county_to_region_map(get_california_counties(), regions)
    cons = load_raw_consumption(paths.COUNTY_CONSUMPTION_XLSX)
    drivers = statewide_drivers(ds, drivers_cache)
    ctx = build_context(ds, gen, cec, regions, cmap, cons, drivers)
    ctx["drivers"] = drivers
    ctx["generation_df"] = gen
    return ctx
