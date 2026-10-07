import numpy as np
import pandas as pd

from componergy.figures.imbalance import bootstrap_raf, compute_imbalance_composites


def test_bootstrap_raf_detects_elevated_risk_with_sensible_ci():
    rng = np.random.default_rng(0)
    normal_vals = rng.normal(0, 1, 200)
    event_vals = rng.normal(1.5, 1, 30)
    thr = np.quantile(normal_vals, 0.90)

    raf, lo, hi, p_e, p_n = bootstrap_raf(event_vals, normal_vals, thr, n_boot=2000, seed=42)

    assert raf > 1.5
    assert lo < raf < hi


def test_compute_imbalance_composites_raf_is_reproducible_across_calls():
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2001-01-01", periods=n, freq="MS")

    imb = pd.Series(rng.normal(2000, 100, n), index=idx)
    heat_mask = pd.Series([False] * n, index=idx)
    heat_mask.iloc[:20] = True
    imb.iloc[:20] += 300

    data = {"imb_gwh": imb, "event_masks": {"Heatwave": heat_mask}, "normal": ~heat_mask}

    result1 = compute_imbalance_composites(data, n_boot=500)
    result2 = compute_imbalance_composites(data, n_boot=500)

    assert (result1["RAF"].values == result2["RAF"].values).all()
    assert (result1["RAF_CI95_lo"].values == result2["RAF_CI95_lo"].values).all()
    assert (result1["RAF_CI95_hi"].values == result2["RAF_CI95_hi"].values).all()


def test_compute_imbalance_composites_fdr_corrects_across_events():
    rng = np.random.default_rng(1)
    n = 200
    idx = pd.date_range("2001-01-01", periods=n, freq="MS")

    imb = pd.Series(rng.normal(2000, 50, n), index=idx)
    heat_mask = pd.Series([False] * n, index=idx)
    heat_mask.iloc[:20] = True
    imb.iloc[:20] += 400

    drought_mask = pd.Series([False] * n, index=idx)
    drought_mask.iloc[20:40] = True
    normal_mask = ~(heat_mask | drought_mask)

    data = {"imb_gwh": imb, "event_masks": {"Heatwave": heat_mask, "Drought": drought_mask}, "normal": normal_mask}

    comp = compute_imbalance_composites(data, n_boot=200)
    heat_row = comp[comp["Event"] == "Heatwave"].iloc[0]
    drought_row = comp[comp["Event"] == "Drought"].iloc[0]

    assert heat_row["q_value"] < 0.05
    assert drought_row["q_value"] > 0.05


def test_bootstrap_raf_matches_manuscript_ratio_minus_one_convention():
    rng = np.random.default_rng(0)
    normal_vals = rng.normal(0, 1, 200)
    event_vals = rng.normal(1.5, 1, 30)
    thr = np.quantile(normal_vals, 0.90)

    raf, lo, hi, p_e, p_n = bootstrap_raf(event_vals, normal_vals, thr, n_boot=500, seed=42)
    assert abs(raf - (p_e / p_n - 1.0)) < 1e-9

    identical_vals = rng.normal(0, 1, 200)
    raf_null, _, _, p_e2, p_n2 = bootstrap_raf(identical_vals, identical_vals, thr, n_boot=500, seed=1)
    assert abs(raf_null) < 0.3
