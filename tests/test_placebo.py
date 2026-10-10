import numpy as np
import pytest
import pandas as pd

from componergy.analysis.placebo import placebo_test, placebo_table, circular_shift_mask


def _masks(n=204):
    ev = np.zeros(n, dtype=bool)
    ev[120:135] = True
    ev[160:175] = True
    return ev, ~ev


def test_circular_shift_preserves_event_count():
    ev, _ = _masks()
    shifted = circular_shift_mask(ev, 37)
    assert shifted.sum() == ev.sum()
    assert not np.array_equal(shifted, ev)


def test_placebo_rejects_a_pure_trend_confound():
    rng = np.random.default_rng(0)
    n = 204
    ev, nm = _masks(n)
    values = np.linspace(0, 3, n) + rng.normal(0, 0.3, n)

    r = placebo_test(values, ev, nm, n_perm=1000)
    assert r["observed_delta_z"] > 0
    assert r["p_value"] > 0.05


def test_placebo_detects_a_genuine_event_response():
    rng = np.random.default_rng(0)
    n = 204
    ev, nm = _masks(n)
    values = rng.normal(0, 0.3, n)
    values[ev] += 2.0

    r = placebo_test(values, ev, nm, n_perm=1000)
    assert r["p_value"] < 0.05
    assert r["observed_delta_z"] > r["placebo_ci_hi"]


def test_placebo_exact_p_value_is_floored_by_the_number_of_distinct_shifts():
    rng = np.random.default_rng(0)
    n = 204
    ev, nm = _masks(n)
    values = rng.normal(0, 0.1, n)
    values[ev] += 50.0
    r = placebo_test(values, ev, nm)
    assert r["n_perm_valid"] == 181
    assert r["p_exact"] == pytest.approx(1 / 182)
    assert r["p_value"] < 1e-6


def test_placebo_uses_each_distinct_shift_at_most_once():
    rng = np.random.default_rng(1)
    ev, nm = _masks(204)
    r = placebo_test(rng.normal(0, 1, 204), ev, nm, n_perm=5000)
    assert r["n_perm_valid"] == 181
    r2 = placebo_test(rng.normal(0, 1, 204), ev, nm, n_perm=40)
    assert r2["n_perm_valid"] == 40


def test_placebo_table_covers_every_column_event_pair_with_fdr():
    rng = np.random.default_rng(0)
    n = 204
    idx = pd.date_range("2008-01-01", periods=n, freq="MS")
    ev, nm = _masks(n)

    df = pd.DataFrame({"A": rng.normal(0, 1, n), "B": rng.normal(0, 1, n)}, index=idx)
    event_masks = {"Heatwave": pd.Series(ev, index=idx), "Drought": pd.Series(np.roll(ev, 60), index=idx)}
    out = placebo_table(df, event_masks, pd.Series(nm, index=idx), n_perm=200)

    assert len(out) == 4
    assert "q_value" in out.columns
