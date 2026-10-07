import numpy as np
import pandas as pd

from componergy.analysis.event_validation import (
    validate_against_known_events, summarize_validation, KNOWN_GRID_EVENTS,
)


def _series(n=204):
    return pd.date_range("2008-01-01", periods=n, freq="MS")


def test_validation_flags_events_when_index_genuinely_spikes():
    idx = _series()
    rng = np.random.default_rng(0)
    risk = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    for m in KNOWN_GRID_EVENTS:
        p = pd.Period(m, freq="M")
        risk[(risk.index.year == p.year) & (risk.index.month == p.month)] += 5.0

    v = validate_against_known_events(risk)
    s = summarize_validation(v)

    assert s["n_in_record"] == len(KNOWN_GRID_EVENTS)
    assert s["hit_rate"] == 1.0
    assert s["hit_rate"] > s["expected_by_chance"]


def test_validation_reports_failure_on_pure_noise():
    idx = _series()
    rng = np.random.default_rng(7)
    risk = pd.Series(rng.normal(0, 1, len(idx)), index=idx)

    s = summarize_validation(validate_against_known_events(risk))
    assert s["hit_rate"] < 1.0


def test_validation_keeps_out_of_record_events_visible():
    idx = pd.date_range("2008-01-01", periods=24, freq="MS")
    risk = pd.Series(np.arange(24, dtype=float), index=idx)

    v = validate_against_known_events(risk)
    assert (~v["In_Record"]).all()
    assert len(v) == len(KNOWN_GRID_EVENTS)
