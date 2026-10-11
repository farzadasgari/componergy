import re
from pathlib import Path

import numpy as np
import pandas as pd

from componergy.analysis.event_definition_check import event_definition_check
from componergy.analysis.events import build_event_masks


def _series(seed=0):
    idx = pd.date_range("1951-01-01", "2026-08-01", freq="MS")
    rng = np.random.default_rng(seed)
    trend = np.linspace(-0.6, 0.6, len(idx))
    mk = lambda off: pd.Series(off + rng.normal(0, 0.9, len(idx)), index=idx)
    return mk(trend), mk(0.0), mk(0.0)


def test_the_check_has_eight_rows_and_uses_only_the_90_10_10_definition(monkeypatch):
    import componergy.analysis.event_definition_check as edc

    seen = []

    def spy(*args, **kwargs):
        out = build_event_masks(*args, **kwargs)
        seen.append(out[2]["quantiles"])
        return out

    monkeypatch.setattr(edc, "build_event_masks", spy)
    sti, sapei, scdhi = _series()
    t = edc.event_definition_check(sti, sapei, scdhi)
    assert len(t) == 2 * 2 * 2
    assert set(t["Baseline"]) == {"full_record", "window"} and set(t["Smoothing_months"]) == {1, 3}
    assert set(t["N_window_months"]) == {204, 296}
    assert len(seen) == 8 and all(q == {"heat": 0.90, "drought": 0.10, "compound": 0.10} for q in seen)


def test_a_window_baseline_gives_about_ten_percent_by_construction():
    sti, sapei, scdhi = _series()
    t = event_definition_check(sti, sapei, scdhi)
    w = t[t["Baseline"] == "window"]
    for col in ("Heatwave_raw", "Drought_raw", "Compound"):
        assert (w[col] < 0.13 * w["N_window_months"]).all()


def test_a_full_record_baseline_gives_a_higher_heat_share_on_a_warming_series():
    sti, sapei, scdhi = _series()
    t = event_definition_check(sti, sapei, scdhi)
    wide = t.pivot_table(index=["Window", "Smoothing_months"], columns="Baseline", values="Share_raw_Heatwave")
    assert (wide["full_record"] > wide["window"]).all()


def test_build_event_masks_defaults_to_the_90_10_10_definition():
    sti, sapei, scdhi = _series()
    _, _, meta = build_event_masks(sti, sapei, scdhi)
    assert meta["quantiles"] == {"heat": 0.90, "drought": 0.10, "compound": 0.10}


def test_download_defaults_follow_the_climate_record_you_actually_have():
    cli = Path("scripts/run_download_pipeline.py").read_text()
    assert re.search(r'--noaa-start-year".*default=1951', cli) and re.search(r'--noaa-end-year".*default=2026', cli)
    reg = Path("src/componergy/download/registry.py").read_text()
    assert "noaa_start_year: int = 1951" in reg and "noaa_end_year: int = 2026" in reg
