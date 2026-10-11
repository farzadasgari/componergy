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
