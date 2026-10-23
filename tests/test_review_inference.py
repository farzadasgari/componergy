import numpy as np
import pandas as pd
import pytest

from componergy.review.freeze import digest, diff_hashes, hash_tree
from componergy.review.inference import (
    pairs_block_bootstrap, season_preserving_shifts, shift_null_generic, whole_year_window,
    year_block_permutation_null,
)
from componergy.analysis.regression import design_matrix, shift_null_coefficients, shift_range


def test_season_preserving_shifts_are_multiples_of_twelve_and_need_whole_years():
    s = season_preserving_shifts(204)
    assert (s % 12 == 0).all() and s[0] == 12 and s[-1] == 192 and len(s) == 16
    assert len(season_preserving_shifts(288)) == 23
    with pytest.raises(ValueError):
        season_preserving_shifts(205)


def test_whole_year_windows():
    idx = pd.date_range("2001-01-01", "2025-08-01", freq="MS")
    assert len(whole_year_window(idx, "calendar")) == 288
    w = whole_year_window(idx, "water")
    assert (w[0], w[-1], len(w)) == (pd.Timestamp("2001-10-01"), pd.Timestamp("2024-09-01"), 276)
    idx2 = pd.date_range("2008-01-01", "2024-12-01", freq="MS")
    assert len(whole_year_window(idx2, "calendar")) == 204 and len(whole_year_window(idx2, "water")) == 192


def test_generic_shift_null_matches_the_original_and_year_permutation_keeps_calendar_months():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", periods=240, freq="MS")
    Z, _ = design_matrix(idx, None, trend="linear")
    D = (rng.random((240, 2)) < 0.1).astype(float)
    y = rng.normal(size=240)
    ref = shift_null_coefficients(y, Z, D)
    got = shift_null_generic(y, Z, D, shift_range(240))
    assert np.allclose(ref, got, atol=1e-10)

    # calendar month of every event is preserved by a block permutation of water years
    win = whole_year_window(idx, "water")
    sel = np.flatnonzero(idx.isin(win))
    blocks = D[sel].reshape(-1, 12, 2)
    rng2 = np.random.default_rng(1)
    Dp = blocks[rng2.permutation(len(blocks))].reshape(-1, 2)
    for j in range(2):
        assert Dp[:, j].sum() == D[sel][:, j].sum()
        assert sorted(idx[sel].month[Dp[:, j] > 0]) == sorted(idx[sel].month[D[sel][:, j] > 0])
    null = year_block_permutation_null(y, Z, D, idx, "water", n_perm=50, seed=42)
    assert null.shape == (50, 2) and np.isfinite(null).all()
    assert np.allclose(null, year_block_permutation_null(y, Z, D, idx, "water", n_perm=50, seed=42))


def test_pairs_bootstrap_covers_truth_and_reports_dropped_draws():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2000-10-01", periods=240, freq="MS")
    x = rng.normal(size=240)
    y = 2.0 * x + rng.normal(size=240)
    X = np.column_stack([np.ones(240), x])
    b = pairs_block_bootstrap(y, X, idx, [1], n_boot=300, seed=42)
    assert b["ci_lo"][0] < 2.0 < b["ci_hi"][0] and b["n_ok"] + b["n_dropped"] == 300
    # an event dummy that is zero on most years makes some draws rank deficient
    d = np.zeros(240)
    d[:12] = 1.0
    b2 = pairs_block_bootstrap(y, np.column_stack([np.ones(240), d]), idx, [1], n_boot=200, seed=42)
    assert b2["n_dropped"] > 0


def test_freeze_hash_is_order_independent_and_detects_one_byte(tmp_path):
    (tmp_path / "figures" / "figure_01").mkdir(parents=True)
    a, b = tmp_path / "figures/figure_01/a.csv", tmp_path / "figures/figure_01/b.csv"
    a.write_bytes(b"abc")
    b.write_bytes(b"def")
    h = hash_tree(tmp_path, ("figures/figure_0*",))
    assert digest(h) == digest(dict(reversed(list(h.items()))))
    b.write_bytes(b"deg")
    h2 = hash_tree(tmp_path, ("figures/figure_0*",))
    assert digest(h) != digest(h2) and diff_hashes(h, h2)["changed"] == ["figures/figure_01/b.csv"]
