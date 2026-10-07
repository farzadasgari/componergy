import numpy as np
import pandas as pd

from componergy.analysis.superadditivity import superadditivity_test, superadditivity_table


def _make_masks(n=400):
    normal = np.zeros(n, dtype=bool);
    normal[:250] = True
    heat = np.zeros(n, dtype=bool);
    heat[250:300] = True
    drought = np.zeros(n, dtype=bool);
    drought[300:350] = True
    compound = np.zeros(n, dtype=bool);
    compound[350:] = True
    return heat, drought, compound, normal


def test_superadditivity_detects_no_interaction_when_effects_are_purely_additive():
    rng = np.random.default_rng(0)
    heat, drought, compound, normal = _make_masks()
    v = rng.normal(0, 0.3, 400)
    v[heat] += 1.0
    v[drought] += 0.5
    v[compound] += 1.5

    r = superadditivity_test(v, heat, drought, compound, normal, n_boot=2000)

    assert r["ci_lo"] < 0 < r["ci_hi"]
    assert r["p_value"] > 0.05
    assert abs(r["additive_expectation"] - 1.5) < 0.15


def test_superadditivity_recovers_known_positive_interaction():
    rng = np.random.default_rng(0)
    heat, drought, compound, normal = _make_masks()
    v = rng.normal(0, 0.3, 400)
    v[heat] += 1.0
    v[drought] += 0.5
    v[compound] += 2.5

    r = superadditivity_test(v, heat, drought, compound, normal, n_boot=2000)

    assert r["interaction"] > 0
    assert r["ci_lo"] > 0
    assert r["p_value"] < 0.05
    assert abs(r["interaction"] - 1.0) < 0.2


def test_superadditivity_recovers_known_negative_interaction():
    rng = np.random.default_rng(0)
    heat, drought, compound, normal = _make_masks()
    v = rng.normal(0, 0.3, 400)
    v[heat] += 1.0
    v[drought] += 0.5
    v[compound] += 0.8

    r = superadditivity_test(v, heat, drought, compound, normal, n_boot=2000)

    assert r["interaction"] < 0
    assert r["ci_hi"] < 0
    assert abs(r["interaction"] - (-0.7)) < 0.2


def test_superadditivity_table_runs_per_column_with_fdr():
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.date_range("2001-01-01", periods=n, freq="MS")
    heat, drought, compound, normal = _make_masks(n)

    strong = rng.normal(0, 0.3, n)
    strong[heat] += 1.0;
    strong[drought] += 0.5;
    strong[compound] += 2.5
    flat = rng.normal(0, 0.3, n)
    flat[heat] += 1.0;
    flat[drought] += 0.5;
    flat[compound] += 1.5

    df = pd.DataFrame({"Strong": strong, "Flat": flat}, index=idx)
    event_masks = {
        "Heatwave": pd.Series(heat, index=idx),
        "Drought": pd.Series(drought, index=idx),
        "Compound": pd.Series(compound, index=idx),
    }
    out = superadditivity_table(df, event_masks, pd.Series(normal, index=idx), n_boot=1000)

    assert len(out) == 2
    assert "q_value" in out.columns
    assert out.set_index("Source").loc["Strong", "q_value"] < 0.05
    assert out.set_index("Source").loc["Flat", "q_value"] > 0.05
