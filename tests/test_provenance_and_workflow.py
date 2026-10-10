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
