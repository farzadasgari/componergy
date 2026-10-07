import json
import tempfile
from pathlib import Path

from componergy.analysis.study_periods import collect_study_periods, _find_period


def test_find_period_locates_nested_metadata():
    payload = {"A": {"B": {"period_start": "2001-01-01", "period_end": "2025-08-01", "n_months": 296}}}
    found = _find_period(payload)
    assert found["period_start"] == "2001-01-01"
    assert found["n_months"] == 296


def test_find_period_returns_none_when_absent():
    assert _find_period({"Overview": {"something": 1}}) is None


def test_collect_study_periods_reads_every_figure_log():
    td = Path(tempfile.mkdtemp())
    (td / "figure_01").mkdir(parents=True)
    (td / "figure_06").mkdir(parents=True)
    json.dump({"Overview": {"period_start": "1951-01-01", "period_end": "2025-08-01", "n_months": 896}},
              open(td / "figure_01" / "figure_01_trend.json", "w"))
    json.dump({"Overview": {"period_start": "2001-01-01", "period_end": "2025-08-01", "n_months": 296}},
              open(td / "figure_06" / "figure_06_imbalance.json", "w"))

    df = collect_study_periods(td)
    assert len(df) == 2
    assert set(df["Figure"]) == {"figure_01_trend", "figure_06_imbalance"}
    assert df.set_index("Figure").loc["figure_01_trend", "period_start"] == "1951-01-01"


def test_collect_study_periods_reports_missing_period_without_failing():
    td = Path(tempfile.mkdtemp())
    (td / "x").mkdir(parents=True)
    json.dump({"Overview": {"no_period": 1}}, open(td / "x" / "nothing.json", "w"))

    df = collect_study_periods(td)
    assert len(df) == 1
    assert df["period_start"].isna().all()


def test_collect_study_periods_skips_malformed_json():
    td = Path(tempfile.mkdtemp())
    (td / "x").mkdir(parents=True)
    open(td / "x" / "broken.json", "w").write("{not valid json")
    json.dump({"Overview": {"period_start": "2008-01-01", "period_end": "2024-12-01", "n_months": 204}},
              open(td / "x" / "good.json", "w"))

    df = collect_study_periods(td)
    assert len(df) == 1
    assert df["Figure"].iloc[0] == "good"
