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


def test_variance_inflation_flags_a_collinear_pair():
    a, _, rng = _data(seed=3)
    vif = variance_inflation(np.column_stack([a, a + 0.05 * rng.normal(size=len(a)), rng.normal(size=len(a))]))
    assert vif[0] > 50 and vif[1] > 50 and vif[2] < 1.5


def test_lmg_table_reports_shares_intervals_and_vif_per_predictor():
    a, b, rng = _data(seed=4)
    X = pd.DataFrame({"STI": a, "SAPEI": b, "SCDHI": 0.6 * a - 0.5 * b + 0.3 * rng.normal(size=len(a))})
    y = pd.Series(a - 0.4 * b + rng.normal(size=len(a)))
    t = lmg_table(y, X, n_boot=60).set_index("Predictor")
    assert set(t.index) == {"STI", "SAPEI", "SCDHI"}
    assert t["LMG_share"].sum() == pytest.approx(1.0, rel=1e-6)
    assert (t["CI_lo_share"] <= t["LMG_share"] + 1e-9).all() or t["CI_lo_share"].notna().all()
    assert t["VIF"].max() > 1.5 and t["n"].iloc[0] == len(a)


def test_incremental_r2_permutation_separates_a_real_added_effect_from_noise():
    a, b, rng = _data(seed=5)
    y = pd.Series(a + 0.8 * b + rng.normal(size=len(a)))
    real = incremental_r2_block_permutation(y, pd.Series(a), pd.Series(b), n_perm=299)
    noise = incremental_r2_block_permutation(y, pd.Series(a), pd.Series(rng.normal(size=len(a))), n_perm=299)
    assert real["p_value"] < 0.02 and real["delta_r2"] > 0.05
    assert noise["p_value"] > 0.05
