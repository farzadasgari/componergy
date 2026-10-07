import numpy as np
import pandas as pd

from componergy.analysis.system_boundary import (
    compute_import_fraction, validate_system_boundary, CA_NET_IMPORT_FRACTION_RANGE,
)


def _gen(n=120):
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(18000, 1500, n), index=idx)


def test_import_fraction_recovers_a_known_injected_share():
    g = _gen()
    d = g / (1 - 0.28)
    out = compute_import_fraction(d, g)
    assert abs(out["Implied_Import_Fraction"].median() - 0.28) < 0.01


def test_validation_accepts_a_realistic_california_import_share():
    g = _gen()
    d = g / (1 - 0.28)
    r = validate_system_boundary(d, g)
    assert r["median_within_expected_range"] is True
    assert r["n_months"] == 120


def test_validation_flags_an_implausibly_small_gap():
    g = _gen()
    d = g * 1.02
    r = validate_system_boundary(d, g)
    assert r["median_within_expected_range"] is False
    assert "BELOW" in r["verdict"]


def test_validation_flags_an_implausibly_large_gap():
    g = _gen()
    d = g * 3.0
    r = validate_system_boundary(d, g)
    assert r["median_within_expected_range"] is False
    assert "ABOVE" in r["verdict"]


def test_validation_reports_months_where_generation_exceeded_demand():
    g = _gen()
    d = g.copy()
    d.iloc[:30] = g.iloc[:30] * 0.8
    r = validate_system_boundary(d, g)
    assert r["fraction_of_months_negative"] > 0.2


def test_validation_handles_no_overlap_gracefully():
    g = _gen()
    d = pd.Series([1.0], index=pd.to_datetime(["1990-01-01"]))
    r = validate_system_boundary(d, g)
    assert r["n_months"] == 0
    assert r["verdict"] == "no overlapping data"
