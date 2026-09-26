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
