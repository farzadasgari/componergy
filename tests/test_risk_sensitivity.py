import numpy as np
import pandas as pd

from componergy.analysis.risk_sensitivity import (
    build_risk_variants,
    compare_risk_variants,
    double_calm_share,
    REFERENCE,
)


def _idx(n=120):
    return pd.date_range("2010-01-01", periods=n, freq="MS")


def test_a_double_calm_month_scores_as_positive_risk_under_the_product_but_not_the_alternatives():
    idx = _idx(4)
    g = pd.Series([-2.0, 2.0, -1.0, 0.5], index=idx)
    d = pd.Series([-2.0, 2.0, 1.0, -0.5], index=idx)
    v = build_risk_variants(g, d)
    assert v[REFERENCE].iloc[0] == 4.0
    assert v["product of positive parts (stress only)"].iloc[0] == 0.0
    assert v["additive (z-sum)"].iloc[0] == -4.0
    assert v[REFERENCE].iloc[1] == 4.0
    assert v["product of positive parts (stress only)"].iloc[1] == 4.0


def test_double_calm_share_counts_positive_months_with_two_negative_factors():
    idx = _idx(4)
    g = pd.Series([-2.0, 2.0, -1.0, 1.0], index=idx)
    d = pd.Series([-2.0, 2.0, 1.0, 1.0], index=idx)
    r = double_calm_share(g, d)
    assert r["n_positive_risk_months"] == 3
    assert r["n_positive_from_both_negative"] == 1
    assert abs(r["share_of_positive_months_that_are_double_calm"] - 1 / 3) < 1e-12


def test_identical_variants_agree_perfectly_and_share_the_peak_month():
    rng = np.random.default_rng(0)
    idx = _idx()
    g = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    d = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    v = build_risk_variants(g, d)
    v["copy"] = v[REFERENCE]
    cmp = compare_risk_variants(v).set_index("Variant")
    assert cmp.loc["copy", "Spearman_vs_reference"] == 1.0
    assert bool(cmp.loc["copy", "Same_peak_month_as_reference"]) is True
    assert cmp.loc["copy", "Top10_overlap_with_reference"] == 10
    assert cmp.loc[REFERENCE, "Rank_of_reference_peak_month"] == 1
