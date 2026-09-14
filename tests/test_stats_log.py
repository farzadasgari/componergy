import json
from pathlib import Path
import tempfile

from componergy.figures.stats_log import write_figure_log


def test_log_and_json_are_both_written():
    test_dir = Path(tempfile.mkdtemp())
    sections = {"Overview": "test figure", "Stats": {"mean": 0.5, "n": 100}}
    write_figure_log(test_dir / "figure_01", sections)

    assert (test_dir / "figure_01.log").exists()
    assert (test_dir / "figure_01.json").exists()


def test_json_round_trips_exactly():
    test_dir = Path(tempfile.mkdtemp())
    sections = {"A": {"x": 1, "y": 2.5}, "B": "plain text section"}
    write_figure_log(test_dir / "out", sections)

    with open(test_dir / "out.json") as f:
        loaded = json.load(f)
    assert loaded == sections


def test_log_text_contains_section_titles_and_key_values():
    test_dir = Path(tempfile.mkdtemp())
    write_figure_log(test_dir / "out", {"My Section": {"mean": 0.5}})

    text = (test_dir / "out.log").read_text()
    assert "---- My Section ----" in text
    assert "mean: 0.5" in text
