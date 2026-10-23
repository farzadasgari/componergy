import hashlib

import pytest

from componergy.provenance import (
    sha256_file,
    directory_fingerprint,
    describe_path,
    build_manifest,
    compare_manifests,
    package_versions,
)
from componergy.workflow import plan, STAGES, STAGE_NAMES


def test_sha256_matches_hashlib_and_changes_with_content(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    assert sha256_file(f) == hashlib.sha256(b"hello").hexdigest()
    f.write_bytes(b"hellO")
    assert sha256_file(f) != hashlib.sha256(b"hello").hexdigest()


def test_directory_fingerprint_changes_when_a_file_is_added_removed_or_truncated(tmp_path):
    (tmp_path / "x.nc").write_bytes(b"1234567890")
    base = directory_fingerprint(tmp_path)
    (tmp_path / "y.nc").write_bytes(b"1")
    assert directory_fingerprint(tmp_path)["listing_sha256"] != base["listing_sha256"]
    (tmp_path / "y.nc").unlink()
    assert directory_fingerprint(tmp_path) == base
    (tmp_path / "x.nc").write_bytes(b"12345")
    assert directory_fingerprint(tmp_path)["listing_sha256"] != base["listing_sha256"]


def test_a_missing_path_is_recorded_as_missing_not_an_error(tmp_path):
    assert describe_path(tmp_path / "nope.xlsx") == {"path": str(tmp_path / "nope.xlsx"), "exists": False}


def test_manifest_records_environment_seeds_inputs_and_outputs_and_comparison_finds_a_changed_input(tmp_path):
    a = tmp_path / "in.csv"
    a.write_text("v1")
    m1 = build_manifest({"input": a}, {"missing_output": tmp_path / "o.nc"}, root=tmp_path)
    assert {"created_utc", "code", "python", "platform", "packages", "seeds", "inputs", "outputs"} <= set(m1)
    assert m1["inputs"]["input"]["sha256"] == hashlib.sha256(b"v1").hexdigest()
    assert m1["outputs"]["missing_output"]["exists"] is False
    assert m1["seeds"]["bootstrap_raf"] == 42
    assert package_versions(("numpy", "definitely_not_a_package"))["definitely_not_a_package"] is None
    a.write_text("v2")
    m2 = build_manifest({"input": a}, {"missing_output": tmp_path / "o.nc"}, root=tmp_path)
    diff = compare_manifests(m1, m2)
    assert list(diff["changed_files"]) == ["inputs.input"]


def test_stage_names_are_unique_and_every_stage_points_at_a_real_script():
    from pathlib import Path
    assert len(set(STAGE_NAMES)) == len(STAGE_NAMES)
    for s in STAGES:
        assert Path("scripts", f"{s.script}.py").exists(), s.script


def test_plan_orders_stages_selects_ranges_and_passes_flags_only_where_accepted():
    p = plan(start="figure-02", end="figure-06", baseline="window", demand_source="eia861m", exists=lambda n: True)
    assert [s["stage"] for s in p] == ["figure-02", "figure-03", "figure-04", "figure-05", "figure-06"]
    cmd = {s["stage"]: s["command"] for s in p}
    assert "--baseline" in cmd["figure-04"] and "--demand-source" not in cmd["figure-04"]
    assert "--demand-source" in cmd["figure-06"] and "--baseline" not in cmd["figure-03"]


def test_plan_blocks_or_skips_stages_whose_inputs_are_missing():
    missing_cec = lambda n: n != "COUNTY_CONSUMPTION_XLSX"
    blocked = {s["stage"]: s["status"] for s in plan(only=["figure-04", "figure-02"], exists=missing_cec)}
    assert blocked["figure-04"].startswith("BLOCKED") and blocked["figure-02"] == "run"
    skipped = {s["stage"]: s["status"] for s in plan(only=["figure-04"], skip_missing=True, exists=missing_cec)}
    assert skipped["figure-04"].startswith("skip")


def test_plan_rejects_an_unknown_stage():
    with pytest.raises(ValueError, match="unknown stage"):
        plan(only=["figure-99"], exists=lambda n: True)


def test_robustness_stage_runs_after_figure_07_and_before_supplementary():
    names = [s.name for s in STAGES if not s.optional]
    assert names.index("figure-07") + 1 == names.index("robustness") == names.index("supplementary") - 1
    p = plan(start="figure-07", end="supplementary", baseline="window", exists=lambda n: True)
    assert [s["stage"] for s in p] == ["figure-07", "robustness", "supplementary"]
    assert "--baseline" not in p[1]["command"]  # it runs both baselines itself


def test_review_v8_is_an_optional_stage_after_robustness_that_needs_its_inputs():
    names = list(STAGE_NAMES)
    assert names.index("review-v8") == names.index("robustness") + 1
    default = [s["stage"] for s in plan(exists=lambda n: True)]
    assert "review-v8" not in default
    assert [s["stage"] for s in plan(start="figure-07", end="supplementary", exists=lambda n: True)] == \
        ["figure-07", "robustness", "supplementary"]
    assert [s["stage"] for s in plan(only=["review-v8"], exists=lambda n: True)] == ["review-v8"]
    assert [s["stage"] for s in plan(start="robustness", end="review-v8", exists=lambda n: True)] == \
        ["robustness", "review-v8"]
    blocked = plan(only=["review-v8"], exists=lambda n: n != "COUNTY_CONSUMPTION_XLSX")[0]
    assert blocked["status"].startswith("BLOCKED") and "COUNTY_CONSUMPTION_XLSX" in blocked["status"]
    listed = plan(only=["review-v8"], exists=lambda n: False)
    assert listed[0]["stage"] == "review-v8" and listed[0]["status"].startswith("BLOCKED")
