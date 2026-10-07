import numpy as np

from componergy.analysis.block_bootstrap import (
    lag1_autocorrelation, suggest_block_length, moving_block_indices, block_bootstrap_raf,
)


def _ar1(n, phi, rng):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, 1)
    return x


def test_lag1_autocorrelation_recovers_known_ar1_coefficient():
    rng = np.random.default_rng(0)
    x = _ar1(4000, 0.6, rng)
    assert abs(lag1_autocorrelation(x) - 0.6) < 0.05


def test_suggest_block_length_grows_with_autocorrelation():
    rng = np.random.default_rng(0)
    weak = _ar1(204, 0.1, rng)
    strong = _ar1(204, 0.8, rng)
    assert suggest_block_length(strong) > suggest_block_length(weak)


def test_moving_block_indices_are_contiguous_and_correct_length():
    rng = np.random.default_rng(0)
    idx = moving_block_indices(100, 10, rng)
    assert len(idx) == 100
    first_block = idx[:10]
    assert np.all(np.diff(first_block) == 1)
    assert idx.max() < 100


def test_moving_block_with_block_one_reduces_to_iid():
    rng = np.random.default_rng(0)
    idx = moving_block_indices(50, 1, rng)
    assert len(idx) == 50
    assert idx.max() < 50


def test_block_bootstrap_raf_gives_wider_ci_than_iid_on_autocorrelated_data():
    rng = np.random.default_rng(0)
    n = 204
    x = _ar1(n, 0.6, rng)

    ev = np.zeros(n, dtype=bool)
    ev[40:60] = True
    ev[120:140] = True
    nm = ~ev
    thr = np.quantile(x[nm], 0.90)

    r_block = block_bootstrap_raf(x, ev, nm, thr, n_boot=1200, seed=1)
    r_iid = block_bootstrap_raf(x, ev, nm, thr, block=1, n_boot=1200, seed=1)

    assert r_block["block"] > 1
    assert (r_block["ci_hi"] - r_block["ci_lo"]) > (r_iid["ci_hi"] - r_iid["ci_lo"])


def test_block_bootstrap_raf_uses_ratio_minus_one_convention():
    rng = np.random.default_rng(0)
    n = 204
    x = rng.normal(0, 1, n)
    ev = np.zeros(n, dtype=bool)
    ev[:40] = True
    nm = ~ev
    thr = np.quantile(x[nm], 0.90)
    r = block_bootstrap_raf(x, ev, nm, thr, n_boot=300, seed=1)
    assert abs(r["raf"] - (r["p_event"] / r["p_normal"] - 1.0)) < 1e-9
