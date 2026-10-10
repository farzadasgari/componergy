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
