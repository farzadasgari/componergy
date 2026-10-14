import numpy as np
import pandas as pd

from componergy.analysis.events import build_event_masks
from componergy.analysis.robustness import COLUMNS, MDE80_FACTOR, family_rows, shift_summary


def _inputs(seed=0):
    idx = pd.date_range("1990-01-01", "2020-12-01", freq="MS")
    rng = np.random.default_rng(seed)
    mk = lambda: pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    sti, sapei, scdhi = mk(), mk(), mk()
    window = idx[idx >= "2005-01-01"]
    season = 10 * np.sin(2 * np.pi * window.month.values / 12)
    raw = pd.DataFrame({"A": 100 + season + rng.normal(0, 2, len(window)),
                        "B": 50 + rng.normal(0, 1, len(window))}, index=window)
    return sti, sapei, scdhi, window, raw


def test_robustness_csv_has_the_schema_and_leaves_undefined_cells_nan(tmp_path):
    sti, sapei, scdhi, window, raw = _inputs()
    masks, normal, meta = build_event_masks(sti, sapei, scdhi, window_index=window, smooth_months=3)
    rows, shifts = family_rows(raw, masks, normal, meta, "synthetic", "full_record", 3)

    few = dict(masks)
    few["Compound"] = pd.Series(False, index=window)
    few["Compound"].iloc[[10, 50]] = True  # below the 3-month minimum: compound effect and A undefined
    rows_few, _ = family_rows(raw, few, normal, meta, "synthetic", "window", 1)

    path = tmp_path / "robustness_long.csv"
    pd.concat([rows, rows_few], ignore_index=True).to_csv(path, index=False)
    back = pd.read_csv(path)

    assert tuple(back.columns) == COLUMNS
    assert len(back) == 4 and list(back["outcome"]) == ["A", "B", "A", "B"]
    assert set(back["baseline"]) == {"full_record", "window"} and set(back["smoothing"]) == {1, 3}
    ok = back.iloc[:2]
    assert ok[["effect_pct", "effect_z", "p_exact", "q_bh", "A", "SE_A", "MDE80"]].notna().all().all()
    assert np.allclose(ok["MDE80"], MDE80_FACTOR * ok["SE_A"])
    assert (ok["A_ci_lo"] <= ok["A"]).all() and (ok["A"] <= ok["A_ci_hi"]).all()
    assert (ok["p_exact"] >= 1.0 / (shifts["S_effect"] + 1).values).all()
    bad = back.iloc[2:]
    assert bad[["effect_pct", "effect_z", "p_exact", "A", "A_p", "A_q", "SE_A", "MDE80"]].isna().all().all()

    summary = shift_summary(shifts)
    assert np.allclose(summary["p_floor_at_S_max"], 1.0 / (summary["S_max"] + 1))
