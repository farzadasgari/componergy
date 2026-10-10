import re
from pathlib import Path

import numpy as np
import pandas as pd

from componergy.analysis.event_definition_check import event_definition_check, MANUSCRIPT_EVENT_COUNTS
from componergy.analysis.events import build_event_masks


def _series(seed=0):
    idx = pd.date_range("1951-01-01", "2026-08-01", freq="MS")
    rng = np.random.default_rng(seed)
    trend = np.linspace(-0.6, 0.6, len(idx))
    mk = lambda off: pd.Series(off + rng.normal(0, 0.9, len(idx)), index=idx)
    return mk(trend), mk(0.0), mk(0.0)


def test_the_check_covers_every_combination_for_each_quoted_window():
    sti, sapei, scdhi = _series()
    t = event_definition_check(sti, sapei, scdhi)
    assert len(t) == 2 * 2 * 2 * 2
    assert set(t["Baseline"]) == {"full_record", "window"} and set(t["Smoothing_months"]) == {1, 3}
    assert set(t["Quantiles"]) == {"final 90/10/10", "exploratory 85/15/12"}
    assert set(t["N_window_months"]) == {204, 296}
