"""Out-of-sample sanity check"""

from __future__ import annotations

import numpy as np
import pandas as pd

KNOWN_GRID_EVENTS = {
    "2020-08": "CAISO rotating outages during the August 2020 West-wide heat wave",
    "2020-09": "September 2020 record heat and statewide Flex Alerts",
    "2022-09": "September 2022 heat dome; CAISO record demand and emergency alerts",
    "2021-07": "July 2021 Bootleg Fire / drought-constrained hydro and transfer limits",
    "2024-07": "July 2024 sustained statewide heat wave",
}


def validate_against_known_events(risk_series: pd.Series, known_events=None,
                                  high_risk_q=0.90) -> pd.DataFrame:
    """Locate each known grid-emergency month in the risk series and report where it ranks."""
    if known_events is None:
        known_events = KNOWN_GRID_EVENTS

    s = risk_series.dropna()
    if len(s) == 0:
        return pd.DataFrame(columns=["Month", "Description", "In_Record", "Risk_Value",
                                     "Percentile", "Rank", "Flagged_High_Risk"])

    thr = float(np.quantile(s.values, high_risk_q))
    ranks = s.rank(ascending=False, method="min")

    rows = []
    for month_str, description in sorted(known_events.items()):
        period = pd.Period(month_str, freq="M")
        matches = s.index[(s.index.year == period.year) & (s.index.month == period.month)]

        if len(matches) == 0:
            rows.append({"Month": month_str, "Description": description, "In_Record": False,
                         "Risk_Value": np.nan, "Percentile": np.nan, "Rank": np.nan,
                         "Flagged_High_Risk": False})
            continue

        ts = matches[0]
        value = float(s.loc[ts])
        pct = float((s.values <= value).mean() * 100.0)
        rows.append({"Month": month_str, "Description": description, "In_Record": True,
                     "Risk_Value": value, "Percentile": pct, "Rank": int(ranks.loc[ts]),
                     "Flagged_High_Risk": bool(value >= thr)})

    return pd.DataFrame(rows)


def summarize_validation(validation_df: pd.DataFrame, high_risk_q=0.90) -> dict:
    """Aggregate the per-event table into a hit rate against the chance baseline."""
    present = validation_df[validation_df["In_Record"]]
    n_present = len(present)
    if n_present == 0:
        return {"n_known_events": len(validation_df), "n_in_record": 0, "n_flagged": 0,
                "hit_rate": np.nan, "expected_by_chance": 1.0 - high_risk_q,
                "median_percentile": np.nan}

    n_flagged = int(present["Flagged_High_Risk"].sum())
    return {
        "n_known_events": int(len(validation_df)),
        "n_in_record": int(n_present),
        "n_flagged": n_flagged,
        "hit_rate": float(n_flagged / n_present),
        "expected_by_chance": float(1.0 - high_risk_q),
        "median_percentile": float(present["Percentile"].median()),
    }
