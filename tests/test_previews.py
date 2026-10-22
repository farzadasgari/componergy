import os

import numpy as np
from PIL import Image

from componergy.figures.previews import build_previews
from componergy.provenance import directory_fingerprint
from componergy.workflow import STAGE_NAMES


def _source(path, size=(1200, 600)):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path, dpi=(600, 600))


def test_preview_is_downscaled_by_one_sixth_with_100_dpi_and_source_is_unchanged(tmp_path):
    src = tmp_path / "figure_01" / "a.jpg"
    _source(src)
    before = src.read_bytes()
    res = build_previews(tmp_path)
    dst = tmp_path / "previews" / "figure_01" / "a.jpg"
    assert res["written"] == [dst] and not list(tmp_path.rglob("*.tmp"))
    with Image.open(dst) as im:
        assert im.size == (200, 100) and im.format == "JPEG"
        assert tuple(round(d) for d in im.info["dpi"]) == (100, 100)
    assert src.read_bytes() == before


def test_a_newer_preview_is_skipped_unless_forced_and_previews_are_not_resampled_again(tmp_path):
    src = tmp_path / "a.jpg"
    _source(src)
    build_previews(tmp_path)
    dst = tmp_path / "previews" / "a.jpg"
    os.utime(dst, (src.stat().st_mtime + 10,) * 2)
    marker = dst.read_bytes()
    res = build_previews(tmp_path)
    assert res["written"] == [] and res["skipped"] == [dst] and dst.read_bytes() == marker
    assert build_previews(tmp_path, force=True)["written"] == [dst]
    assert not (tmp_path / "previews" / "previews").exists()


def test_previews_stage_is_last_and_excluded_from_directory_fingerprints(tmp_path):
    assert STAGE_NAMES[-1] == "previews"
    (tmp_path / "x.csv").write_text("1")
    base = directory_fingerprint(tmp_path, exclude=("previews",))
    (tmp_path / "previews").mkdir()
    (tmp_path / "previews" / "a.jpg").write_bytes(b"jpg")
    assert directory_fingerprint(tmp_path, exclude=("previews",)) == base
