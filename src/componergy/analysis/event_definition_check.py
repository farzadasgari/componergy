from __future__ import annotations

import pandas as pd

from componergy.analysis.events import build_event_masks

MANUSCRIPT_EVENT_COUNTS = {
    ("2008-01-01", "2024-12-01"): {"n_months": 204, "Heatwave": 30, "Drought": 21, "Compound": 16},
    ("2001-01-01", "2025-08-01"): {"n_months": 296, "Heatwave": 66, "Drought": 46, "Compound": 40},
}
QUANTILE_SETS = {"final 90/10/10": (0.90, 0.10, 0.10), "exploratory 85/15/12": (0.85, 0.15, 0.12)}
BASELINES = ("full_record", "window")
SMOOTHINGS = (1, 3)


def event_definition_check(sti: pd.Series, sapei: pd.Series, scdhi: pd.Series,
                           reference: dict | None = None) -> pd.DataFrame:
    """One row per (window, baseline, smoothing, quantile set) with counts and distance to the reference."""
    reference = MANUSCRIPT_EVENT_COUNTS if reference is None else reference
    rows = []
    for (start, end), ref in reference.items():
        window = sti.index[(sti.index >= start) & (sti.index <= end)]
        if len(window) == 0:
            continue
        for baseline in BASELINES:
            for smooth in SMOOTHINGS:
                for qname, (hq, dq, cq) in QUANTILE_SETS.items():
                    _, _, meta = build_event_masks(sti, sapei, scdhi, window_index=window, smooth_months=smooth,
                                                   baseline=baseline, heat_q=hq, drought_q=dq, compound_q=cq)
                    got = meta["counts_raw"]
                    dist = sum(abs(got[e] - ref[e]) for e in ("Heatwave", "Drought", "Compound"))
                    rows.append({"Window": f"{start[:7]} to {end[:7]}", "N_window_months": len(window),
                                 "Baseline": baseline, "Smoothing_months": smooth, "Quantiles": qname,
                                 "Heatwave": got["Heatwave"], "Drought": got["Drought"], "Compound": got["Compound"],
                                 "Manuscript_Heatwave": ref["Heatwave"], "Manuscript_Drought": ref["Drought"],
                                 "Manuscript_Compound": ref["Compound"], "Distance": int(dist)})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["Window", "Distance"]).reset_index(drop=True)
        out["Best_for_window"] = out.groupby("Window")["Distance"].transform("min") == out["Distance"]
    return out
