from __future__ import annotations

import pandas as pd

from componergy.analysis.events import build_event_masks

WINDOWS = (("2008-01-01", "2024-12-01"), ("2001-01-01", "2025-08-01"))
BASELINES = ("full_record", "window")
SMOOTHINGS = (1, 3)
EVENTS = ("Heatwave", "Drought", "Compound")


def event_definition_check(sti: pd.Series, sapei: pd.Series, scdhi: pd.Series) -> pd.DataFrame:
    """Event-count sensitivity of the 90/10/10 definition: one row per (window, baseline, smoothing)."""
    rows = []
    for start, end in WINDOWS:
        window = sti.index[(sti.index >= start) & (sti.index <= end)]
        if len(window) == 0:
            continue
        for baseline in BASELINES:
            for smooth in SMOOTHINGS:
                _, _, meta = build_event_masks(sti, sapei, scdhi, window_index=window, smooth_months=smooth,
                                               baseline=baseline)
                raw, excl, ep, thr = (meta["counts_raw"], meta["counts_mutually_exclusive"],
                                      meta["episodes"], meta["thresholds"])
                n = meta["n_months"]
                rows.append({
                    "Window": f"{start[:7]} to {end[:7]}", "N_window_months": n,
                    "Baseline": meta["baseline"], "Smoothing_months": meta["smooth_months"],
                    "Heatwave_raw": raw["Heatwave"], "Drought_raw": raw["Drought"], "Compound": raw["Compound"],
                    "Heatwave_exclusive": excl["Heatwave"], "Drought_exclusive": excl["Drought"],
                    "Episodes_Heatwave": ep["Heatwave_raw"], "Episodes_Drought": ep["Drought_raw"],
                    "Episodes_Compound": ep["Compound"],
                    **{f"Share_raw_{e}": raw[e] / n for e in EVENTS},
                    "Threshold_STI_heat": thr["STI_heat"], "Threshold_SAPEI_drought": thr["SAPEI_drought"],
                    "Threshold_SCDHI_compound": thr["SCDHI_compound"],
                })
    return pd.DataFrame(rows)
