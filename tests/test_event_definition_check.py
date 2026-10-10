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


def test_the_combination_that_generated_the_reference_is_found_with_zero_distance():
    sti, sapei, scdhi = _series(seed=3)
    win = sti.index[(sti.index >= "2008-01-01") & (sti.index <= "2024-12-01")]
    _, _, meta = build_event_masks(sti, sapei, scdhi, window_index=win, smooth_months=1, baseline="full_record",
                                   heat_q=0.85, drought_q=0.15, compound_q=0.12)
    ref = {("2008-01-01", "2024-12-01"): {"n_months": 204, **meta["counts_raw"]}}
    t = event_definition_check(sti, sapei, scdhi, reference=ref)
    best = t[t["Best_for_window"]].iloc[0]
    assert best["Distance"] == 0
    assert (best["Baseline"], best["Smoothing_months"], best["Quantiles"]) == ("full_record", 1, "exploratory 85/15/12")


def test_a_window_baseline_cannot_reproduce_counts_far_above_ten_percent():
    sti, sapei, scdhi = _series()
    t = event_definition_check(sti, sapei, scdhi)
    w = t[(t["Window"] == "2001-01 to 2025-08") & (t["Baseline"] == "window") & (t["Quantiles"] == "final 90/10/10")]
    assert (w["Heatwave"] < 0.13 * 296).all()
    assert MANUSCRIPT_EVENT_COUNTS[("2001-01-01", "2025-08-01")]["Heatwave"] > 0.2 * 296


def test_download_defaults_follow_the_climate_record_you_actually_have():
    cli = Path("scripts/run_download_pipeline.py").read_text()
    assert re.search(r'--noaa-start-year".*default=1951', cli) and re.search(r'--noaa-end-year".*default=2026', cli)
    reg = Path("src/componergy/download/registry.py").read_text()
    assert "noaa_start_year: int = 1951" in reg and "noaa_end_year: int = 2026" in reg
