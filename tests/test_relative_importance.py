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
