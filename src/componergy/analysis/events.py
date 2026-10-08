"""Heatwave/drought/compound event classification and standardized anomalies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def compute_percentile_threshold(da: xr.DataArray, q: float) -> xr.DataArray:
    """Empirical q-th percentile per grid cell, across the full time record."""
    threshold = da.quantile(q, dim="time", skipna=True)
    return threshold.drop_vars("quantile", errors="ignore")


def classify_heatwave(sti: xr.DataArray, percentile: float = 0.90) -> xr.DataArray:
    """Boolean: STI exceeds its per-cell `percentile`-th percentile."""
    threshold = compute_percentile_threshold(sti, percentile)
    return sti > threshold


def classify_drought(sapei: xr.DataArray, percentile: float = 0.10) -> xr.DataArray:
    """Boolean: SAPEI falls below its per-cell `percentile`-th percentile."""
    threshold = compute_percentile_threshold(sapei, percentile)
    return sapei < threshold


def classify_compound(scdhi: xr.DataArray, percentile: float = 0.10) -> xr.DataArray:
    """Boolean: SCDHI falls below its per-cell `percentile`-th percentile."""
    threshold = compute_percentile_threshold(scdhi, percentile)
    return scdhi < threshold


def classify_events(
        sti: xr.DataArray, sapei: xr.DataArray, scdhi: xr.DataArray,
        heat_percentile: float = 0.90, drought_percentile: float = 0.10, compound_percentile: float = 0.10,
) -> xr.Dataset:
    """Classify all three event types plus mutually-exclusive heat/drought variants."""
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
    """"Normal" = none of the three event types. The more standard baseline choice."""
    return ~(is_heatwave | is_drought | is_compound)


def normal_mask_opposite_extreme(scdhi: xr.DataArray, percentile: float = 0.90) -> xr.DataArray:
    """"Normal" = SCDHI in its own top `percentile`-th percentile (wettest/coolest)."""
    threshold = compute_percentile_threshold(scdhi, percentile)
    return scdhi >= threshold


def compute_delta_z(standardized_var: xr.DataArray, event_mask: xr.DataArray,
                    normal_mask: xr.DataArray) -> xr.DataArray:
    """Standardized anomaly: mean during event_mask minus mean during normal_mask."""
    event_mean = standardized_var.where(event_mask).mean(dim="time", skipna=True)
    normal_mean = standardized_var.where(normal_mask).mean(dim="time", skipna=True)
    return event_mean - normal_mean


EVENT_BASELINES = ("full_record", "window")


def count_event_clusters(mask) -> int:
    """Number of episodes: runs of consecutive True months."""
    m = np.asarray(pd.Series(mask).fillna(False).astype(bool).values)
    if m.size == 0:
        return 0
    starts = m & ~np.concatenate([[False], m[:-1]])
    return int(starts.sum())


def exclusive_event_masks(event_masks: dict) -> dict:
    """Mutually exclusive event categories."""
    if "Compound" not in event_masks:
        return dict(event_masks)
    comp = event_masks["Compound"].fillna(False).astype(bool)
    return {name: (m if name == "Compound" else (m.fillna(False).astype(bool) & ~comp))
            for name, m in event_masks.items()}


def event_overlap_summary(flags: dict) -> dict:
    """How the univariate heat/drought criteria relate to the compound index, in months."""
    heat, drought, compound = flags["heat"], flags["drought"], flags["compound"]
    return {
        "heat_and_drought_raw": int((heat & drought).sum()),
        "compound_total": int(compound.sum()),
        "compound_with_heat_criterion": int((compound & heat).sum()),
        "compound_with_drought_criterion": int((compound & drought).sum()),
        "compound_with_both_criteria": int((compound & heat & drought).sum()),
        "compound_with_neither_criterion": int((compound & ~heat & ~drought).sum()),
        "heat_only_also_compound": int((heat & ~drought & compound).sum()),
        "drought_only_also_compound": int((drought & ~heat & compound).sum()),
    }


def build_event_masks(sti, sapei, scdhi, window_index=None, smooth_months=3,
                      baseline="full_record", heat_q=0.90, drought_q=0.10, compound_q=0.10,
                      return_flags=False):
    if baseline not in EVENT_BASELINES:
        raise ValueError(f"baseline must be one of {EVENT_BASELINES}, got {baseline!r}")

    window_index = pd.DatetimeIndex(sti.index if window_index is None else window_index)

    def smooth(s):
        return s.astype(float).rolling(smooth_months, center=True, min_periods=1).mean()

    sm = {"sti": smooth(sti), "sapei": smooth(sapei), "scdhi": smooth(scdhi)}
    in_window = {k: s.reindex(window_index) for k, s in sm.items()}
    reference = sm if baseline == "full_record" else in_window

    thr_heat = float(np.nanquantile(reference["sti"].values, heat_q))
    thr_drought = float(np.nanquantile(reference["sapei"].values, drought_q))
    thr_compound = float(np.nanquantile(reference["scdhi"].values, compound_q))

    heat = in_window["sti"] > thr_heat
    drought = in_window["sapei"] < thr_drought
    compound = in_window["scdhi"] < thr_compound
    valid = in_window["sti"].notna() & in_window["sapei"].notna() & in_window["scdhi"].notna()

    heat_only = heat & ~drought
    drought_only = drought & ~heat
    normal = valid & ~(heat | drought | compound)

    event_masks = {"Heatwave": heat_only, "Drought": drought_only, "Compound": compound}
    excl = exclusive_event_masks(event_masks)
    flags = {"heat": heat, "drought": drought, "compound": compound, "valid": valid}
    meta = {
        "baseline": baseline,
        "smooth_months": int(smooth_months),
        "window_start": str(window_index.min().date()),
        "window_end": str(window_index.max().date()),
        "n_months": int(len(window_index)),
        "quantiles": {"heat": heat_q, "drought": drought_q, "compound": compound_q},
        "thresholds": {"STI_heat": thr_heat, "SAPEI_drought": thr_drought, "SCDHI_compound": thr_compound},
        "counts_raw": {"Heatwave": int(heat.sum()), "Drought": int(drought.sum()), "Compound": int(compound.sum())},
        "counts_exclusive": {k: int(v.sum()) for k, v in event_masks.items()},
        "counts_mutually_exclusive": {k: int(v.sum()) for k, v in excl.items()},
        "episodes": {
            "Heatwave_raw": count_event_clusters(heat), "Drought_raw": count_event_clusters(drought),
            "Compound": count_event_clusters(compound),
            "Heatwave_exclusive": count_event_clusters(excl["Heatwave"]),
            "Drought_exclusive": count_event_clusters(excl["Drought"]),
        },
        "overlap": event_overlap_summary(flags),
        "n_normal": int(normal.sum()),
    }
    if return_flags:
        return event_masks, normal, meta, flags
    return event_masks, normal, meta


def event_counts_by_baseline(sti, sapei, scdhi, window_index=None, smooth_months=3,
                             heat_q=0.90, drought_q=0.10, compound_q=0.10) -> pd.DataFrame:
    rows = []
    for baseline in EVENT_BASELINES:
        _, _, meta = build_event_masks(sti, sapei, scdhi, window_index=window_index,
                                       smooth_months=smooth_months, baseline=baseline,
                                       heat_q=heat_q, drought_q=drought_q, compound_q=compound_q)
        for event in ("Heatwave", "Drought", "Compound"):
            rows.append({
                "Baseline": baseline, "Event": event,
                "Window": f"{meta['window_start']} to {meta['window_end']}",
                "N_window_months": meta["n_months"],
                "N_raw": meta["counts_raw"][event],
                "N_exclusive": meta["counts_mutually_exclusive"][event],
                "Episodes": meta["episodes"].get(f"{event}_raw", meta["episodes"].get(event)),
                "Share_raw": meta["counts_raw"][event] / meta["n_months"],
                "Threshold": meta["thresholds"][{"Heatwave": "STI_heat", "Drought": "SAPEI_drought",
                                                 "Compound": "SCDHI_compound"}[event]],
            })
    return pd.DataFrame(rows)
