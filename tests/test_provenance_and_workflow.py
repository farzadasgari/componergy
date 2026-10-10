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
