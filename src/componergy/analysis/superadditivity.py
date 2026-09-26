"""
Formal test for super-additivity (nonlinear amplification) of
compound drought-heat events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SAMPLE = 3


def _delta_z(values, event_mask, normal_mask):
    """Mean anomaly during event months minus mean during normal months; NaN if either sample is too small."""
    ev = values[event_mask]
    nm = values[normal_mask]
    ev = ev[np.isfinite(ev)]
    nm = nm[np.isfinite(nm)]
    if len(ev) < MIN_SAMPLE or len(nm) < MIN_SAMPLE:
        return np.nan
    return float(np.mean(ev) - np.mean(nm))
