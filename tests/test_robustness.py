import json

import numpy as np
import pandas as pd

from componergy.analysis.events import build_event_masks
from componergy.analysis.robustness import (
    COLUMNS, FAMILY_TRENDS, MDE80_FACTOR, compare_row, consistency_section, family_rows, shift_summary,
)


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
    rows, shifts = family_rows(raw, masks, normal, meta, "synthetic", "full_record", 3, "linear")

    few = dict(masks)
    few["Compound"] = pd.Series(False, index=window)
    few["Compound"].iloc[[10, 50]] = True  # below the 3-month minimum: compound effect and A undefined
    rows_few, _ = family_rows(raw, few, normal, meta, "synthetic", "window", 1, "quadratic")

    path = tmp_path / "robustness_long.csv"
    pd.concat([rows, rows_few], ignore_index=True).to_csv(path, index=False)
    back = pd.read_csv(path)

    assert tuple(back.columns) == COLUMNS and back.columns[-1] == "trend"
    assert list(back["trend"]) == ["linear", "linear", "quadratic", "quadratic"]
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


def test_trend_column_follows_the_trend_and_each_family_uses_its_figure_trend():
    sti, sapei, scdhi, window, raw = _inputs(seed=1)
    masks, normal, meta = build_event_masks(sti, sapei, scdhi, window_index=window, smooth_months=3)
    lin, _ = family_rows(raw, masks, normal, meta, "synthetic", "full_record", 3, "linear")
    quad, _ = family_rows(raw, masks, normal, meta, "synthetic", "full_record", 3, "quadratic")
    assert set(lin["trend"]) == {"linear"} and set(quad["trend"]) == {"quadratic"}
    assert not np.allclose(lin["effect_z"], quad["effect_z"])  # the trend really reaches the fit
    assert FAMILY_TRENDS == {"imbalance": "linear", "regional_demand": "linear", "generation_share": "quadratic"}


def test_compare_row_ok_within_tolerance_and_mismatch_outside_it():
    ours = {"effect_pct": 1.0, "q_bh": 0.5, "A": np.nan, "n_compound": 63}
    ok = compare_row(ours, {"effect_pct": 1.0 + 5e-10, "q_bh": 0.5 * (1 + 5e-7), "A": np.nan, "n_compound": 63})
    assert ok["status"] == "OK" and ok["failed"] == [] and ok["max_abs_diff"] < 1e-6
    bad = compare_row(ours, {"effect_pct": 1.001, "q_bh": 0.5, "A": 0.2, "n_compound": 62})
    assert bad["status"] == "MISMATCH" and bad["failed"] == ["effect_pct", "A", "n_compound"]
    assert bad["max_abs_diff"] == np.inf  # NaN on one side only


def _write_figure_02(d, rows, smooth_months=3, drop=None):
    """Minimal figure_02 outputs carrying the values of `rows` (generation-share robustness rows)."""
    d.mkdir(parents=True)
    comp = pd.DataFrame({"Mode": "adjusted_share", "Event": "Compound", "Source": rows["outcome"],
                         "Delta_pct": rows["effect_pct"], "Delta_z": rows["effect_z"], "p_exact": rows["p_exact"],
                         "q_value": rows["q_bh"]})
    comp = pd.concat([comp, comp.assign(Mode="total_share", Delta_pct=99.0)], ignore_index=True)
    amp = pd.DataFrame({"Source": rows["outcome"], "Amplification_contrast": rows["A"], "q_value": rows["A_q"]})
    if drop:
        comp = comp.drop(columns=[drop])
    comp.to_csv(d / "figure_02_composite.csv", index=False)
    amp.to_csv(d / "figure_02_amplification.csv", index=False)
    meta = {"baseline": "full_record", "smooth_months": smooth_months,
            "counts_raw": {"Compound": int(rows["n_compound"].iloc[0])}}
    (d / "figure_02_generation_response.json").write_text(json.dumps({"Event definition (shared)": meta}))


def test_consistency_section_reports_ok_mismatch_and_skipped(tmp_path):
    sti, sapei, scdhi, window, raw = _inputs(seed=2)
    masks, normal, meta = build_event_masks(sti, sapei, scdhi, window_index=window, smooth_months=3)
    rows, _ = family_rows(raw, masks, normal, meta, "generation_share", "full_record", 3, "quadratic")
    other, _ = family_rows(raw, masks, normal, meta, "generation_share", "window", 1, "quadratic")
    table = pd.concat([rows, other.assign(effect_pct=123.0)], ignore_index=True)

    _write_figure_02(tmp_path / "ok" / "figure_02", rows)
    lines, counts = consistency_section(table, tmp_path / "ok")
    assert counts == {"OK": 2, "MISMATCH": 0, "SKIPPED": 2}  # figure_04 and figure_06 are absent
    assert sum("generation_share | " in ln and ln.endswith("OK") for ln in lines) == 2
    assert any("regional_demand vs figure_04: SKIPPED (missing" in ln for ln in lines)

    _write_figure_02(tmp_path / "bad" / "figure_02", rows.assign(q_bh=rows["q_bh"] + 0.01))
    lines, counts = consistency_section(table, tmp_path / "bad")
    assert counts["MISMATCH"] == 2 and counts["OK"] == 0
    assert all("q_bh:" in ln for ln in lines if "MISMATCH" in ln)

    _write_figure_02(tmp_path / "nocol" / "figure_02", rows, drop="Delta_z")
    lines, counts = consistency_section(table, tmp_path / "nocol")
    assert counts["OK"] == 2 and any("no matching figure column for effect_z (Delta_z" in ln for ln in lines)

    _write_figure_02(tmp_path / "sm1" / "figure_02", rows, smooth_months=1)
    lines, _ = consistency_section(table, tmp_path / "sm1")
    assert any("generation_share vs figure_02: SKIPPED (figure_02 was built with baseline=full_record, "
               "smoothing=1)" in ln for ln in lines)
