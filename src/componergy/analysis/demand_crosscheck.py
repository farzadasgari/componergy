from __future__ import annotations

import numpy as np
import pandas as pd

from componergy.figures.composite_stats import deseasonalize_monthly

AGREE_CLOSE_CORR = 0.95
AGREE_MODERATE_CORR = 0.85
MIN_OVERLAP_MONTHS = 24


def compare_demand_sources(a_mwh: pd.Series, b_mwh: pd.Series,
                           name_a: str = "CEC", name_b: str = "EIA-861M") -> dict:
    common = a_mwh.dropna().index.intersection(b_mwh.dropna().index)
    if len(common) < MIN_OVERLAP_MONTHS:
        return {"source_a": name_a, "source_b": name_b, "n_months": int(len(common)),
                "verdict": f"fewer than {MIN_OVERLAP_MONTHS} overlapping months; no meaningful comparison"}

    a = a_mwh.loc[common].astype(float)
    b = b_mwh.loc[common].astype(float)
    ratio = a / b
    anom_a, anom_b = deseasonalize_monthly(a), deseasonalize_monthly(b)

    corr_raw = float(np.corrcoef(a.values, b.values)[0, 1])
    corr_anom = float(np.corrcoef(anom_a.values, anom_b.values)[0, 1])
    rms_pct = float(np.sqrt(np.mean(((anom_a / a.mean()) - (anom_b / b.mean())) ** 2)) * 100.0)

    if corr_anom >= AGREE_CLOSE_CORR:
        verdict = (f"close agreement (anomaly correlation {corr_anom:.2f}): the choice of demand source "
                   "cannot change the event composites materially")
    elif corr_anom >= AGREE_MODERATE_CORR:
        verdict = (f"moderate agreement (anomaly correlation {corr_anom:.2f}): composites may differ at "
                   "the margin; report results for both sources")
    else:
        verdict = (f"weak agreement (anomaly correlation {corr_anom:.2f}): the two series measure "
                   "materially different things, so the source choice must be justified explicitly and "
                   "results shown for both")

    return {"source_a": name_a, "source_b": name_b, "n_months": int(len(common)),
            "period_start": str(common.min().date()), "period_end": str(common.max().date()),
            "level_ratio_mean": float(ratio.mean()), "level_ratio_std": float(ratio.std(ddof=1)),
            "corr_raw_series": corr_raw, "corr_deseasonalized_anomalies": corr_anom,
            "anomaly_rms_diff_pct_of_mean": rms_pct, "verdict": verdict}
