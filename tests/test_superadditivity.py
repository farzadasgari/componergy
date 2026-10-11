import numpy as np
import pandas as pd

from componergy.analysis.superadditivity import superadditivity_test, superadditivity_table


def _idx(n=204):
    return pd.date_range("2008-01-01", periods=n, freq="MS")


def _mask(idx, months_by_year):
    keys = {(y, m) for y, ms in months_by_year.items() for m in ms}
    return np.array([(t.year, t.month) in keys for t in idx])


def _masks(idx):
    h = _mask(idx, {2009: [6, 7, 8], 2012: [7, 8], 2015: [6, 7, 8], 2018: [7, 8]})
    d = _mask(idx, {2010: [5, 6, 7], 2013: [6, 7], 2016: [5, 6, 7], 2019: [6, 7]})
    c = _mask(idx, {2011: [7, 8, 9], 2014: [7, 8], 2017: [7, 8, 9], 2021: [6, 7, 8]})
    return h, d, c, ~(h | d | c)


def _noise(n, seed, ar=0.4, sd=1.0):
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = ar * e[i - 1] + rng.normal(0, sd)
    return e


def _series(effect_h, effect_d, effect_c, seed=0):
    idx = _idx()
    h, d, c, nm = _masks(idx)
    v = _noise(len(idx), seed)
    v[h] += effect_h
    v[d] += effect_d
    v[c] += effect_c
    return v, h, d, c, nm


def test_no_interaction_is_reported_when_effects_are_purely_additive():
    v, h, d, c, nm = _series(2.0, 1.0, 3.0)
    r = superadditivity_test(v, h, d, c, nm, n_boot=400)
    assert r["ci_lo"] < 0 < r["ci_hi"]
    assert r["p_value"] > 0.05
    assert abs(r["additive_expectation"] - 3.0) < 1.5


def test_positive_interaction_is_recovered_and_flagged():
    v, h, d, c, nm = _series(2.0, 1.0, 7.0)
    r = superadditivity_test(v, h, d, c, nm, n_boot=400)
    assert r["interaction"] > 0 and r["ci_lo"] > 0
    assert r["p_value"] < 0.05 and r["p_exact"] < 0.05
    assert abs(r["interaction"] - 4.0) < 1.5


def test_negative_interaction_is_recovered_and_flagged():
    v, h, d, c, nm = _series(2.0, 1.0, -2.0)
    r = superadditivity_test(v, h, d, c, nm, n_boot=400)
    assert r["interaction"] < 0 and r["ci_hi"] < 0
    assert r["p_value"] < 0.05


def test_exclusive_categories_stop_compound_months_being_subtracted_as_heat():
    idx = _idx()
    h, d, c, nm = _masks(idx)
    h_overlapping = h | c
    v = _noise(len(idx), 3)
    v[h] += 2.0
    v[d] += 1.0
    v[c] += 7.0
    nm_ = ~(h_overlapping | d | c)
    excl = superadditivity_test(v, h_overlapping, d, c, nm_, n_boot=200, exclusive=True)
    naive = superadditivity_test(v, h_overlapping, d, c, nm_, n_boot=200, exclusive=False)
    assert excl["exclusive"] is True and naive["exclusive"] is False
    assert excl["interaction"] > naive["interaction"] + 1.0


def test_result_carries_the_bootstrap_comparison_columns_and_block_ci_is_wider_than_iid():
    v, h, d, c, nm = _series(2.0, 1.0, 3.0, seed=5)
    v = v + _noise(len(v), 6, ar=0.8, sd=0.8)
    r = superadditivity_test(v, h, d, c, nm, n_boot=600)
    for k in ("ci_lo_block", "ci_hi_block", "p_block", "ci_lo_iid", "ci_hi_iid", "p_iid",
              "block_length", "n_boot_valid", "se_null", "n_shifts"):
        assert k in r
    assert r["n_shifts"] == 181
    assert (r["ci_hi_block"] - r["ci_lo_block"]) > (r["ci_hi_iid"] - r["ci_lo_iid"])


def test_nan_result_when_a_category_is_too_small():
    idx = _idx()
    h, d, c, nm = _masks(idx)
    h_tiny = np.zeros(len(idx), bool)
    h_tiny[10] = True
    r = superadditivity_test(_noise(len(idx), 1), h_tiny, d, c, ~(h_tiny | d | c), n_boot=100)
    assert np.isnan(r["interaction"]) and r["n_boot_valid"] == 0


def test_randomization_test_is_calibrated_when_there_is_no_interaction():
    rejections, n_sims = 0, 100
    rng = np.random.default_rng(77)
    for s in range(n_sims):
        v, h, d, c, nm = _series(0.0, 0.0, 0.0, seed=int(rng.integers(1, 10 ** 6)))
        rejections += superadditivity_test(v, h, d, c, nm, n_boot=50, seed=s)["p_exact"] < 0.05
    assert rejections / n_sims < 0.13


def test_table_runs_per_column_with_fdr_and_separates_real_from_null_series():
    idx = _idx()
    h, d, c, nm = _masks(idx)
    strong = _noise(len(idx), 11)
    strong[h] += 2.0
    strong[d] += 1.0
    strong[c] += 11.0
    flat = _noise(len(idx), 12)
    flat[h] += 2.0
    flat[d] += 1.0
    flat[c] += 3.0
    df = pd.DataFrame({"Strong": strong, "Flat": flat}, index=idx)
    masks = {"Heatwave": pd.Series(h, index=idx), "Drought": pd.Series(d, index=idx),
             "Compound": pd.Series(c, index=idx)}
    out = superadditivity_table(df, masks, pd.Series(nm, index=idx), n_boot=200).set_index("Source")
    assert len(out) == 2 and "q_value" in out.columns
    assert out.loc["Strong", "q_value"] < 0.05
    assert out.loc["Flat", "q_value"] > 0.05
