import numpy as np
import pandas as pd

from componergy.review.nullsim import (
    fit_var_simulator, placebo_positions, run_nullsim, simulate_raw, to_ns, wilson_interval, zscore_array,
)


def test_wilson_interval_and_placebo_window_counts():
    lo, hi = wilson_interval(5, 100)
    assert lo < 0.05 < hi
    dates = pd.date_range("1951-01-01", "2026-08-01", freq="MS")
    assert len(placebo_positions(296, dates)) == 26 and len(placebo_positions(204, dates)) == 34


def test_var_simulator_recovers_persistence():
    rng = np.random.default_rng(0)
    T = 908
    dates = pd.date_range("1951-01-01", periods=T, freq="MS")
    a = np.zeros((T, 3))
    for t in range(1, T):
        a[t] = 0.5 * a[t - 1] + rng.normal(size=3)
    raw = a + 0.3 * np.sin(2 * np.pi * dates.month.values / 12)[:, None]
    raw[:2, 1:] = np.nan
    par = fit_var_simulator(raw, dates.month.values - 1)
    assert par["p"] in (1, 2)
    sim = simulate_raw(par, np.random.default_rng(1))
    assert sim.shape == (T, 3) and np.isfinite(sim).all()
    r1 = np.corrcoef(sim[1:, 0], sim[:-1, 0])[0, 1]
    assert 0.35 < r1 < 0.65


def test_nullsim_smoke_runs_and_reports_every_cell():
    rng = np.random.default_rng(3)
    T = 908
    dates = pd.date_range("1951-01-01", periods=T, freq="MS")
    raw = np.column_stack([np.cumsum(rng.normal(size=T)) * 0.0 + rng.normal(size=T) for _ in range(3)])
    z = np.stack([zscore_array(raw[:, i]) for i in range(3)])
    pos = np.arange(600, 804)  # 2001-01 .. 2017-12
    n = len(pos)
    Y = np.column_stack([rng.normal(size=n) + np.sin(np.arange(n)), rng.normal(size=n)])
    win = {"pos": pos, "index_ns": to_ns(dates[pos]), "Y": Y, "names": ["a", "b"], "trends": ["linear", "linear"]}
    out = run_nullsim(z, to_ns(dates), {"w": win}, raw, n_draws_n2=4, n_perm=19, n_jobs=1, chunk=2,
                       min_cond=3, max_total=12)
    assert len(out["rejection"]) == 2 * 2 * 5 * 5 and "cond_wilson_hi" in out["rejection"] and set(out["rejection"]["generator"]) == {"N1", "N2"}
    assert out["n_draws"]["N1"]["w"] == 34 and not out["diagnostics"].empty


def test_conditional_filter_bounds_and_rates():
    from componergy.review.nullsim import conditional_mask, rate_cell
    m, (lo, hi) = conditional_mask([46, 47, 63, 79, 80, np.nan], 63)
    assert (lo, hi) == (47, 79) and m.tolist() == [False, True, True, True, False, False]
    assert conditional_mask([40, 66, 39, 67], 53)[1] == (40, 66)
    r = rate_cell(np.array([0.01, 0.2, 0.04, np.nan, 0.9]))
    assert r["n_draws"] == 4 and r["rate"] == 0.5 and r["wilson_lo"] < 0.5 < r["wilson_hi"]
