"""Collect the study period actually used by each figure into one table."""

from __future__ import annotations

import json
import pandas as pd


def _find_period(obj):
    if isinstance(obj, dict):
        if "period_start" in obj and "period_end" in obj:
            return {"period_start": obj.get("period_start"),
                    "period_end": obj.get("period_end"),
                    "n_months": obj.get("n_months")}
        for value in obj.values():
            found = _find_period(value)
            if found is not None:
                return found
    return None


def collect_study_periods(figures_dir) -> pd.DataFrame:
    rows = []
    for json_path in sorted(pd.Series(list(figures_dir.rglob("*.json")))):
        try:
            with open(json_path) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        period = _find_period(payload)
        rows.append({
            "Figure": json_path.stem,
            "period_start": (period or {}).get("period_start"),
            "period_end": (period or {}).get("period_end"),
            "n_months": (period or {}).get("n_months"),
            "log_file": str(json_path.relative_to(figures_dir)),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("Figure").reset_index(drop=True) if len(df) else df
