import numpy as np
import pandas as pd
import pytest

from componergy.analysis.relative_importance import (
    lmg_decomposition,
    variance_inflation,
    lmg_table,
    incremental_r2_block_permutation,
)


def _data(n=204, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    return a, b, rng


def test_lmg_shares_add_up_to_the_joint_r2():
    a, b, rng = _data()
    y = 1.0 * a + 0.5 * b + rng.normal(size=len(a))
    d = lmg_decomposition(y, np.column_stack([a, b, a + b + 0.1 * rng.normal(size=len(a))]))
    assert d["lmg"].sum() == pytest.approx(d["total_r2"], rel=1e-9)


def test_identical_predictors_split_the_credit_equally():
    a, _, rng = _data(seed=1)
    y = a + rng.normal(size=len(a))
    d = lmg_decomposition(y, np.column_stack([a, a.copy()]))
    assert d["lmg"][0] == pytest.approx(d["lmg"][1], rel=1e-9)


def test_a_noise_predictor_gets_almost_none_of_the_credit_when_others_explain_y():
    a, b, rng = _data(seed=2)
    y = 2 * a + b + 0.5 * rng.normal(size=len(a))
    noise = rng.normal(size=len(a))
    d = lmg_decomposition(y, np.column_stack([a, b, noise]))
    assert d["lmg"][2] < 0.02 * d["total_r2"]
    assert d["lmg"][0] > d["lmg"][1]
