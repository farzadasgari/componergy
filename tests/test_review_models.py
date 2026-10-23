import numpy as np
import pandas as pd
import xarray as xr

from componergy.analysis.events import build_event_masks
from componergy.analysis.regression import event_effects
from componergy.figures.composite_stats import zscore
from componergy.review.models import (
    analyse, build_event_problem, fit_continuous_model, fit_event_model, loyo_cv_rmse,
    substitution_accounting, substitution_ratios,
)
from tests.synthetic_inputs import _ar1, make_generation, make_indices


def _synthetic_events():
    ds = make_indices()
    gen = make_generation()
    s = {k: zscore(ds[v].mean(["lat", "lon"]).to_series()) for k, v in
         (("sti", "sti"), ("sapei", "sapei_3m"), ("scdhi", "scdhi_3m"))}
    masks, normal, _ = build_event_masks(s["sti"], s["sapei"], s["scdhi"], window_index=gen.index)
    y = (gen["Fossil"] / gen["Total"] * 100).rolling(3, center=True, min_periods=1).mean()
    return y, masks, normal


def test_fit_event_model_without_extra_is_event_effects_exactly():
    y, masks, normal = _synthetic_events()
    ref = event_effects(y, masks, normal, trend="quadratic")["effects"]
    got = fit_event_model(y, masks, normal, "quadratic", n_perm=49, n_boot=50)
    pd.testing.assert_frame_equal(got["effects"][list(ref.columns)], ref, check_exact=True)
    # the engine's own primary columns agree with it to rounding
    t = got["table"].set_index("Term")
    for _, r in ref.iterrows():
        if np.isfinite(r["Delta_native"]):
            assert abs(t.loc[r["Event"], "estimate"] - r["Delta_native"]) < 1e-12
            assert abs(t.loc[r["Event"], "p_norm_shift"] - r["p_value"]) < 1e-12
            assert abs(t.loc[r["Event"], "p_exact_shift"] - r["p_exact"]) < 1e-12
            assert abs(t.loc[r["Event"], "p_hac"] - r["p_hac"]) < 1e-12


def test_event_model_with_controls_changes_the_estimate_but_keeps_the_columns():
    y, masks, normal = _synthetic_events()
    h = pd.Series(np.random.default_rng(0).normal(size=len(y)), index=y.index)
    a = fit_event_model(y, masks, normal, "quadratic", n_perm=0, boot=False)
    b = fit_event_model(y, masks, normal, "quadratic", extra={"h": h}, n_perm=0, boot=False)
    assert list(a["effects"].columns)[:3] == list(b["effects"].columns)[:3]
    assert not np.allclose(a["effects"]["Delta_native"], b["effects"]["Delta_native"])


def test_continuous_model_recovers_planted_interaction_and_tests_reject():
    rng = np.random.default_rng(7)
    n = 600
    idx = pd.date_range("1970-01-01", periods=n, freq="MS")
    h = pd.Series(_ar1(n, 0.5, rng), index=idx)
    d = pd.Series(_ar1(n, 0.5, rng), index=idx)
    y = pd.Series(0.5 * h * d + rng.normal(size=n), index=idx)
    out = fit_continuous_model(y, {"h": h, "d": d, "hd": h * d}, "linear", n_perm=499, n_boot=0)
    r = out["table"].set_index("Term").loc["hd"]
    assert abs(r["estimate"] - 0.5) < 0.15
    assert r["p_norm_shift"] < 0.05 and r["p_yearperm"] < 0.05


def test_shift_inference_is_roughly_calibrated_for_autocorrelated_series():
    rng = np.random.default_rng(11)
    n = 240
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    rej = 0
    for _ in range(200):
        y = pd.Series(_ar1(n, 0.5, rng), index=idx)
        m = pd.Series(_ar1(n, 0.5, rng) > 1.28, index=idx)
        prob = build_event_problem(y, {"Compound": m}, ~m, "linear")
        p = analyse(prob, windows=False)["table"].set_index("Term").loc["Compound", "p_norm_shift"]
        rej += p < 0.05
    assert 0.01 <= rej / 200 <= 0.12


def test_substitution_ratio_recovers_planted_replacement():
    rng = np.random.default_rng(5)
    n = 276
    idx = pd.date_range("2001-10-01", periods=n, freq="MS")
    c = pd.Series(False, index=idx)
    c.iloc[[20, 21, 22, 90, 91, 92, 93, 150, 151, 152, 200, 201]] = True
    hydro = 1000 + 5 * np.sin(np.arange(n)) - 200 * c.values + rng.normal(0, 10, n)
    fossil = 3000 - 0.5 * (hydro - 1000) + rng.normal(0, 10, n)
    Y = pd.DataFrame({"Fossil": fossil, "Hydro": hydro, "Total": fossil + hydro}, index=idx)
    out = substitution_accounting(Y, {"Compound": c}, ~c, "linear", n_boot=200)
    assert abs(out["R_fossil"] - 0.5) < 0.1
    assert out["R_fossil_lo"] < out["R_fossil"] < out["R_fossil_hi"]
    assert substitution_ratios(50.0, -100.0, -50.0) == {"R_fossil": 0.5, "R_unreplaced": 0.5}


def test_loyo_cv_prefers_the_true_model():
    rng = np.random.default_rng(2)
    n = 240
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    x = rng.normal(size=n)
    y = 1.0 * x + rng.normal(size=n) * 0.5
    ones = np.ones((n, 1))
    assert loyo_cv_rmse(y, lambda i: np.column_stack([ones, x]), idx) < loyo_cv_rmse(y, lambda i: ones, idx)


def test_exclude_mask_removes_months_from_the_event_problem():
    y, masks, normal = _synthetic_events()
    yr = y.index.year.values
    full = build_event_problem(y, masks, normal, "linear")
    cut = build_event_problem(y, masks, normal, "linear", exclude=(yr == 2010))
    assert cut.sample.sum() < full.sample.sum() and not cut.sample[yr == 2010].any()
