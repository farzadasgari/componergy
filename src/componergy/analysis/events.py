from __future__ import annotations

import numpy as np
import xarray as xr


def compute_percentile_threshold(da: xr.DataArray, q: float) -> xr.DataArray:
    threshold = da.quantile(q, dim="time", skipna=True)
    return threshold.drop_vars("quantile", errors="ignore")


def classify_heatwave(sti: xr.DataArray, percentile: float = 0.90) -> xr.DataArray:
    threshold = compute_percentile_threshold(sti, percentile)
    return sti > threshold


def classify_drought(sapei: xr.DataArray, percentile: float = 0.10) -> xr.DataArray:
    threshold = compute_percentile_threshold(sapei, percentile)
    return sapei < threshold


def classify_compound(scdhi: xr.DataArray, percentile: float = 0.10) -> xr.DataArray:
    threshold = compute_percentile_threshold(scdhi, percentile)
    return scdhi < threshold


def classify_events(
        sti: xr.DataArray, sapei: xr.DataArray, scdhi: xr.DataArray,
        heat_percentile: float = 0.90, drought_percentile: float = 0.10, compound_percentile: float = 0.10,
) -> xr.Dataset:
    heat = classify_heatwave(sti, heat_percentile)
    drought = classify_drought(sapei, drought_percentile)
    compound = classify_compound(scdhi, compound_percentile)

    heat_only = heat & ~drought
    drought_only = drought & ~heat

    return xr.Dataset({
        "is_heatwave": heat,
        "is_drought": drought,
        "is_compound": compound,
        "is_heatwave_only": heat_only,
        "is_drought_only": drought_only,
    })


def normal_mask_exclude_all_events(is_heatwave: xr.DataArray, is_drought: xr.DataArray,
                                   is_compound: xr.DataArray) -> xr.DataArray:
    return ~(is_heatwave | is_drought | is_compound)


def normal_mask_opposite_extreme(scdhi: xr.DataArray, percentile: float = 0.90) -> xr.DataArray:
    threshold = compute_percentile_threshold(scdhi, percentile)
    return scdhi >= threshold


def compute_delta_z(standardized_var: xr.DataArray, event_mask: xr.DataArray,
                    normal_mask: xr.DataArray) -> xr.DataArray:
    event_mean = standardized_var.where(event_mask).mean(dim="time", skipna=True)
    normal_mean = standardized_var.where(normal_mask).mean(dim="time", skipna=True)
    return event_mean - normal_mean
