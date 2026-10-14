"""
Robustness of the compound effect and the amplification contrast to the event baseline and smoothing.

Rows are built from the existing estimators (adjusted_composite_table, amplification_table); nothing
here re-estimates anything. One row per outcome: the compound effect and the amplification contrast
A = compound - (heat-only + drought-only), with the 80%-power minimum detectable |A| at alpha = 0.05.
Each family is fitted with the trend term of its main figure, and the baseline=full_record, smoothing=3
rows can be checked against that figure's CSVs (compare_row, figure_reference).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from componergy.analysis.regression import adjusted_composite_table, amplification_table
from componergy.figures.composite_stats import benjamini_hochberg_fdr
from componergy.figures.generation import TREND as GENERATION_TREND
from componergy.figures.imbalance import TREND as IMBALANCE_TREND
from componergy.figures.zonal_consumption import TREND as ZONAL_TREND

BASELINES = ("full_record", "window")
SMOOTHINGS = (1, 3)
FAMILY_TRENDS = {"imbalance": IMBALANCE_TREND, "regional_demand": ZONAL_TREND,
                 "generation_share": GENERATION_TREND}
MDE80_FACTOR = 2.8016  # z_{0.975} + z_{0.80}: two-sided alpha = 0.05, power = 0.80
COLUMNS = ("baseline", "smoothing", "family", "outcome", "n_heat", "n_drought", "n_compound",
           "n_compound_exclusive", "effect_pct", "effect_z", "p_exact", "q_bh", "A", "A_ci_lo", "A_ci_hi",
           "A_p", "A_q", "SE_A", "MDE80", "trend")

# The main figures are built with the full-record baseline and 3-month smoothing.
REFERENCE_BASELINE, REFERENCE_SMOOTHING = "full_record", 3
ATOL, RTOL = 1e-9, 1e-6
# Where each family's main-figure numbers live. The CEC-period imbalance is figure 06's primary output.
FIGURE_REFERENCES = {
    "generation_share": {"dir": "figure_02", "composite": "figure_02_composite.csv",
                         "amplification": "figure_02_amplification.csv", "log": "figure_02_generation_response.json",
                         "label": "Source", "amp_label": "Source", "select": {"Mode": "adjusted_share"}},
    "regional_demand": {"dir": "figure_04", "composite": "figure_04_event_composites.csv",
                        "amplification": "figure_04_amplification.csv", "log": "figure_04_zonal_consumption.json",
                        "label": "Region", "amp_label": "Region", "select": {}},
    "imbalance": {"dir": "figure_06", "composite": "figure_06_event_metrics.csv",
                  "amplification": "figure_06_amplification.csv", "log": "figure_06_imbalance.json",
                  "label": None, "amp_label": "Metric", "select": {}, "demand_source": "cec"},
}
# robustness column -> (figure table, figure column); n_compound is the event definition's raw compound count
FIGURE_COLUMNS = {"n_compound": ("log", "counts_raw.Compound"), "effect_pct": ("composite", "Delta_pct"),
                  "effect_z": ("composite", "Delta_z"), "p_exact": ("composite", "p_exact"),
                  "q_bh": ("composite", "q_value"), "A": ("amplification", "Amplification_contrast"),
                  "A_q": ("amplification", "q_value")}


def family_rows(raw_df: pd.DataFrame, event_masks: dict, normal, meta: dict, family: str,
                baseline: str, smoothing: int, trend: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robustness rows for one outcome family, and the calendar-shift count behind each outcome's p-values."""
    comp = adjusted_composite_table(raw_df, event_masks, normal, trend=trend)
    comp["q_value"] = benjamini_hochberg_fdr(comp["p_value"].values)  # within family, all outcome x event
    amp = amplification_table(raw_df, event_masks, normal, trend=trend, label="outcome").set_index("outcome")
    counts = meta["counts_raw"]

    rows, shifts = [], []
    for col in raw_df.columns:
        c = comp[(comp["Source"] == col) & (comp["Event"] == "Compound")].iloc[0]
        a = amp.loc[col]
        rows.append({
            "baseline": baseline, "smoothing": int(smoothing), "family": family, "outcome": col,
            "n_heat": counts["Heatwave"], "n_drought": counts["Drought"], "n_compound": counts["Compound"],
            "n_compound_exclusive": int(c["N_event"]),
            "effect_pct": c["Delta_pct"], "effect_z": c["Delta_z"], "p_exact": c["p_exact"], "q_bh": c["q_value"],
            "A": a["Amplification_contrast"], "A_ci_lo": a["CI_lo"], "A_ci_hi": a["CI_hi"],
            "A_p": a["p_value"], "A_q": a["q_value"], "SE_A": a["SE"], "MDE80": MDE80_FACTOR * a["SE"],
            "trend": trend,
        })
        shifts.append({"baseline": baseline, "smoothing": int(smoothing), "family": family, "outcome": col,
                       "S_effect": int(c["N_shifts"]), "S_A": int(a["N_shifts"])})
    return pd.DataFrame(rows, columns=list(COLUMNS)), pd.DataFrame(shifts)


def shift_summary(shifts: pd.DataFrame) -> pd.DataFrame:
    """Per baseline (and smoothing and family): range of shift counts S and the exact-p floor 1/(S+1)."""
    keys = ["baseline", "smoothing", "family"]
    s = shifts.melt(id_vars=keys, value_vars=["S_effect", "S_A"], value_name="S")
    s = s[s["S"] > 0]  # 0 = estimate undefined, so no randomization was run
    out = s.groupby(keys, sort=False)["S"].agg(S_min="min", S_max="max")
    out["p_floor_at_S_min"] = 1.0 / (out["S_min"] + 1)
    out["p_floor_at_S_max"] = 1.0 / (out["S_max"] + 1)
    return out.reset_index()


def compare_row(ours: dict, theirs: dict, atol: float = ATOL, rtol: float = RTOL) -> dict:
    """
    Compare one robustness row with the figure's numbers, metric by metric (np.isclose: |a-b| <= atol + rtol*|b|).

    Both NaN counts as equal; NaN on one side only is a mismatch with an infinite difference.
    """
    diffs, failed = {}, []
    for metric, b in theirs.items():
        a, b = float(ours[metric]), float(b)
        if np.isnan(a) and np.isnan(b):
            d, ok = 0.0, True
        elif np.isnan(a) or np.isnan(b):
            d, ok = np.inf, False
        else:
            d, ok = abs(a - b), abs(a - b) <= atol + rtol * abs(b)
        diffs[metric] = d
        if not ok:
            failed.append(metric)
    return {"max_abs_diff": max(diffs.values()) if diffs else np.nan, "diffs": diffs, "failed": failed,
            "status": "MISMATCH" if failed else "OK"}


def figure_reference(family: str, figures_dir: Path) -> dict:
    """
    The main figure's value of each FIGURE_COLUMNS metric, per outcome, read from its CSVs and JSON log.

    Returns {"skipped": reason} when a file is missing or the figure was built with another event baseline,
    smoothing or demand source; otherwise {"values": {outcome: {metric: value}}, "missing": [...]}, where
    "missing" names every metric whose figure column was not found (those metrics are not compared).
    """
    spec = FIGURE_REFERENCES[family]
    d = Path(figures_dir) / spec["dir"]
    paths = {k: d / spec[k] for k in ("composite", "amplification", "log")}
    absent = [p.name for p in paths.values() if not p.exists()]
    if absent:
        return {"skipped": f"missing {', '.join(absent)} in {d}"}

    log = json.loads(paths["log"].read_text(encoding="utf-8"))
    event_def = next((v for k, v in log.items() if k.startswith("Event definition")), None)
    if event_def is None:
        return {"skipped": f"no 'Event definition' section in {paths['log'].name}"}
    built = (event_def.get("baseline"), event_def.get("smooth_months"))
    if built != (REFERENCE_BASELINE, REFERENCE_SMOOTHING):
        return {"skipped": f"{spec['dir']} was built with baseline={built[0]}, smoothing={built[1]}"}
    if "demand_source" in spec and log.get("Overview", {}).get("demand_source") != spec["demand_source"]:
        return {"skipped": f"{spec['dir']} was built with demand source "
                           f"{log.get('Overview', {}).get('demand_source')!r}, not {spec['demand_source']!r}"}

    comp = pd.read_csv(paths["composite"])
    for k, v in spec["select"].items():
        comp = comp[comp[k] == v]
    comp = comp[comp["Event"] == "Compound"]
    amp = pd.read_csv(paths["amplification"])
    if spec["label"] is None:  # single-outcome figure: its CSV carries no outcome column
        comp = comp.assign(_outcome=amp[spec["amp_label"]].iloc[0])
    else:
        comp = comp.rename(columns={spec["label"]: "_outcome"})
    tables = {"composite": comp.set_index("_outcome"), "amplification": amp.set_index(spec["amp_label"])}

    values, missing = {}, []
    for metric, (table, column) in FIGURE_COLUMNS.items():
        if table == "log":
            key, sub = column.split(".")
            if sub not in event_def.get(key, {}):
                missing.append(f"{metric} ({column} in {paths['log'].name})")
                continue
            for outcome in tables["composite"].index:
                values.setdefault(outcome, {})[metric] = event_def[key][sub]
        elif column not in tables[table].columns:
            missing.append(f"{metric} ({column} in {paths[table].name})")
        else:
            for outcome, v in tables[table][column].items():
                values.setdefault(outcome, {})[metric] = v
    return {"values": values, "missing": missing}


def consistency_section(table: pd.DataFrame, figures_dir: Path) -> tuple[list[str], dict]:
    """Log lines comparing the baseline=full_record, smoothing=3 rows with the main figures, and the status counts."""
    ref_rows = table[(table["baseline"] == REFERENCE_BASELINE) & (table["smoothing"] == REFERENCE_SMOOTHING)]
    lines = [f"compared: {', '.join(FIGURE_COLUMNS)} (q_bh vs the figure's q_value); "
             f"OK when |a-b| <= {ATOL:g} + {RTOL:g}*|b| for every metric",
             f"rows: baseline={REFERENCE_BASELINE}, smoothing={REFERENCE_SMOOTHING}"]
    counts = {"OK": 0, "MISMATCH": 0, "SKIPPED": 0}
    for family, spec in FIGURE_REFERENCES.items():
        rows = ref_rows[ref_rows["family"] == family]
        ref = figure_reference(family, figures_dir)
        if "skipped" in ref:
            lines.append(f"{family} vs {spec['dir']}: SKIPPED ({ref['skipped']})")
            counts["SKIPPED"] += 1
            continue
        if ref["missing"]:
            lines.append(f"{family} vs {spec['dir']}: no matching figure column for "
                         f"{'; '.join(ref['missing'])}; not compared")
        for _, row in rows.iterrows():
            theirs = ref["values"].get(row["outcome"])
            if theirs is None:
                lines.append(f"  {family} | {row['outcome']}: SKIPPED (no Compound row for this outcome in "
                             f"{spec['dir']})")
                counts["SKIPPED"] += 1
                continue
            res = compare_row(row.to_dict(), theirs)
            detail = "".join(f"; {m}: {row[m]!r} vs {theirs[m]!r}" for m in res["failed"])
            lines.append(f"  {family} | {row['outcome']}: max|diff| = {res['max_abs_diff']:.3g} "
                         f"{res['status']}{detail}")
            counts[res["status"]] += 1
    return lines, counts
