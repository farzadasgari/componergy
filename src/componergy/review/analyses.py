# src/componergy/review/analyses.py
"""The B1, B2 and B3 analysis tables of review v8. Each function returns DataFrames; the script writes them."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from componergy.analysis.regression import adjusted_composite_table, design_matrix, hac_contrast
from componergy.figures.composite_stats import benjamini_hochberg_fdr
from componergy.review.data import GROUP_FAMILY, GROUP_UNIT, HEADLINE
from componergy.review.episodes import compound_episodes, effective_event_counts, episode_water_years
from componergy.review.inference import season_preserving_shifts, water_year_labels
from componergy.review.models import (
    P_TYPES, analyse, build_continuous_problem, build_event_problem, fit_continuous_model, fit_event_model,
    incremental_r2_blocks, loyo_cv_rmse, ols_r2, share_decomposition, substitution_accounting,
)

CHOICE_ORDER = ("p_norm_shift", "p_yearperm", "p_exact_shift", "p_exact_season")
EVENTS = ("Heatwave", "Drought", "Compound")
SEED = 42
D_LEVEL = 1.28  # 10th percentile of the drought driver d = -SAPEI_sm (z units)
INFERENCE_COLS = ["p_norm_shift", "p_exact_shift", "p_exact_season", "p_yearperm", "p_hac", "CI_boot_lo",
                  "CI_boot_hi", "n_boot_ok", "n_boot_dropped", "est_season", "est_water"]


@dataclass(frozen=True)
class Cfg:
    n_perm: int = 1999
    n_boot: int = 2000
    n_r2: int = 1999
    n_draws: int = 1000
    n_perm_sim: int = 499
    n_jobs: int = 1
    seed: int = SEED

    def quick(self) -> "Cfg":
        return replace(self, n_perm=max(self.n_perm // 10, 19), n_boot=max(self.n_boot // 10, 20),
                       n_r2=max(self.n_r2 // 10, 19), n_draws=max(self.n_draws // 10, 5),
                       n_perm_sim=max(self.n_perm_sim // 10, 19))


def chosen_p(row_family, table: pd.DataFrame, choice: dict) -> pd.Series:
    """Per-row p-value of the family's chosen p-type (NaN where no p-type is calibrated)."""
    out = []
    for fam, (_, r) in zip(row_family, table.iterrows()):
        pt = (choice.get("families", {}).get(fam, {}) or {}).get("chosen")
        out.append(r[pt] if pt else np.nan)
    return pd.Series(out, index=table.index, dtype=float)


def _bh(p: pd.Series) -> pd.Series:
    return pd.Series(benjamini_hochberg_fdr(p.values.astype(float)), index=p.index)


def fit(o: dict, ctx: dict, cfg: Cfg, extra=None, trend=None, exclude=None, windows=True, boot=True) -> dict:
    w = ctx["windows"][o["window"]]
    prob = build_event_problem(o["y"], w["masks"], w["normal"], trend or o["trend"], extra=extra, exclude=exclude)
    return analyse(prob, n_perm=cfg.n_perm, n_boot=cfg.n_boot, seed=cfg.seed, boot=boot, windows=windows)


def _series(ctx, names=None, groups=None):
    return [o for o in ctx["outcomes"] if (names is None or o["name"] in names)
            and (groups is None or o["group"] in groups)]


# -------------------------------------------------------------------------------------------- B3
def b3_inference_comparison(ctx, cfg: Cfg, choice: dict) -> tuple:
    """Headline series x (Compound, Heatwave, Drought, A) x every p-type, same point estimate; plus checks."""
    rows, chk, dropped = [], [], 0
    for o in _series(ctx, HEADLINE):
        w = ctx["windows"][o["window"]]
        res = fit_event_model(o["y"], w["masks"], w["normal"], o["trend"], n_perm=cfg.n_perm, n_boot=cfg.n_boot,
                              seed=cfg.seed)
        t = res["table"]
        ref = res["event_effects"]["effects"].set_index("Event")
        for _, r in t.iterrows():
            rows.append({"Series": o["name"], "Group": o["group"], "Family": o["family"], "Term": r["Term"],
                         "N": r["N"], "estimate": r["estimate"], "unit": o["unit"], "CI_shift_lo": r["ci_lo"],
                         "CI_shift_hi": r["ci_hi"], **{c: r[c] for c in INFERENCE_COLS}})
            if r["Term"] in ref.index:
                chk.append(max(abs(r["estimate"] - ref.loc[r["Term"], "Delta_native"]),
                               abs(r["p_norm_shift"] - ref.loc[r["Term"], "p_value"]),
                               abs(r["p_exact_shift"] - ref.loc[r["Term"], "p_exact"]),
                               abs(r["p_hac"] - ref.loc[r["Term"], "p_hac"])))
        dropped += int(t["n_boot_dropped"].max())
    tab = pd.DataFrame(rows)
    tab["p_chosen"] = chosen_p(tab["Family"], tab, choice)
    for pt in list(P_TYPES) + ["p_chosen"]:
        tab["q_" + pt.replace("p_", "", 1)] = tab.groupby("Term")[pt].transform(_bh)
    return tab, {"max_abs_diff_vs_event_effects": float(np.nanmax(chk)), "boot_draws_dropped": dropped}


def effective_events(ctx) -> tuple:
    eff, eps = [], []
    for wname in ("generation", "demand"):
        w = ctx["windows"][wname]
        for ev in EVENTS:
            m = pd.Series(np.asarray(w["masks"][ev], bool), index=w["index"])
            eff.append({"window": wname, "event": ev, **effective_event_counts(m)})
        m = pd.Series(np.asarray(w["masks"]["Compound"], bool), index=w["index"])
        ep = compound_episodes(m)
        ep.insert(0, "window", wname)
        ep["water_year_first"], ep["water_year_last"] = zip(*episode_water_years(ep)) if len(ep) else ([], [])
        eps.append(ep)
    return pd.DataFrame(eff), pd.concat(eps, ignore_index=True)


def p_floors(ctx) -> dict:
    """Smallest attainable p per family and p-type (shift and season types have a floor 1/(S+1))."""
    from componergy.analysis.regression import shift_range
    from componergy.review.inference import whole_year_window

    out = {}
    for fam in ("generation", "demand", "imbalance"):
        idx = ctx["windows"][fam]["index"]
        n_cal = len(whole_year_window(idx, "calendar"))
        out[fam] = {"p_norm_shift": 0.0, "p_exact_shift": 1.0 / (len(shift_range(len(idx))) + 1),
                    "p_exact_season": 1.0 / (len(season_preserving_shifts(n_cal)) + 1)}
    return out


def choose_inference(rejection: pd.DataFrame, floors: dict, n_perm_main: int, alpha: float = 0.05) -> dict:
    """Apply the pre-registered B3 rule family by family (generation, demand, imbalance)."""
    fam_series = {}
    for o_name in HEADLINE:
        fam_series[o_name] = ("generation" if "share" in o_name or "GWh" in o_name else
                              "imbalance" if o_name == "Imbalance" else "demand")
    out = {"alpha": alpha, "rule": ("Amendment 1: calibrated for a series x term if the N2 unconditional AND the N2 "
                                    "conditional (compound count within +-25% of observed) Wilson upper bounds are "
                                    "<= 0.075; N1 is descriptive and not used; family p-type = first calibrated in order "
                                    + ", ".join(CHOICE_ORDER) + "; p-types whose smallest attainable p exceeds "
                                    "alpha are ineligible; none -> effect sizes with bootstrap CIs, no "
                                    "significance claim"), "families": {}}
    for fam in ("generation", "demand", "imbalance"):
        series = [s for s, f in fam_series.items() if f == fam]
        sub = rejection[rejection["series"].isin(series)]
        info = {"series": series, "calibrated": {}, "failing_cells": {}, "eligible": {}, "chosen": None}
        for pt in CHOICE_ORDER:
            floor = floors[fam].get(pt)
            if pt == "p_yearperm":
                floor = 1.0 / (n_perm_main + 1)
            info["eligible"][pt] = bool(floor is None or floor <= alpha)
            cells = sub[sub["p_type"] == pt]
            n2 = cells[cells["generator"] == "N2"].set_index(["series", "term"])
            ok = (n2["wilson_hi"] <= 0.075) & (n2["cond_wilson_hi"] <= 0.075)
            info["calibrated"][pt] = bool(ok.all()) if len(ok) else False
            info["failing_cells"][pt] = [f"{a}|{b}" for (a, b), v in ok.items() if not v]
        for pt in CHOICE_ORDER:
            if info["calibrated"][pt] and info["eligible"][pt]:
                info["chosen"] = pt
                break
        out["families"][fam] = info
    return out


# -------------------------------------------------------------------------------------------- B1
def b1_event_table(ctx, cfg: Cfg, choice: dict) -> tuple:
    rows = []
    for group in ("G-share", "G-energy", "D", "I"):
        raw = ctx["raw_frames"][group]
        grp = _series(ctx, groups=[group])
        w = ctx["windows"][grp[0]["window"]]
        comp = adjusted_composite_table(raw, w["masks"], w["normal"], trend=grp[0]["trend"]).set_index(["Source", "Event"])
        for o in grp:
            t = fit(o, ctx, cfg).get("table").set_index("Term")
            for ev in EVENTS:
                if (o["col"], ev) not in comp.index or ev not in t.index:
                    continue
                c, e = comp.loc[(o["col"], ev)], t.loc[ev]
                rows.append({
                    "Series": o["name"], "Group": group, "Family": o["family"], "Event": ev,
                    "N_event": int(c["N_event"]), "Delta_native": c["Delta_native"], "Delta_unit": GROUP_UNIT[group],
                    "Delta_pct": c["Delta_pct"], "Delta_z": c["Delta_z"], "CI_lo_native": c["CI_lo_native"],
                    "CI_hi_native": c["CI_hi_native"], "CI_boot_lo": e["CI_boot_lo"], "CI_boot_hi": e["CI_boot_hi"],
                    "p_norm_shift": c["p_value"], "p_exact_shift": c["p_exact"], "p_exact_season": e["p_exact_season"],
                    "p_yearperm": e["p_yearperm"], "p_hac": c["p_hac"], "n_boot_dropped": e["n_boot_dropped"],
                    "chk_engine_vs_figure": max(abs(e["estimate"] - c["Delta_native"]),
                                                abs(e["p_norm_shift"] - c["p_value"]),
                                                abs(e["p_exact_shift"] - c["p_exact"]))})
    tab = pd.DataFrame(rows)
    tab["q_norm_shift"] = tab.groupby("Group")["p_norm_shift"].transform(_bh)
    pc = chosen_p(tab["Family"], tab, choice)
    tab["q_chosen"] = pc.groupby(tab["Group"]).transform(_bh).where(pc.notna())
    return tab, {"boot_draws_dropped": int(tab["n_boot_dropped"].max())}


def _driver_summary(ctx) -> pd.DataFrame:
    rows = []
    for wname in ("generation", "demand"):
        w = ctx["windows"][wname]
        reg = w["reg"]
        masks = {k: np.asarray(v, bool) for k, v in w["masks"].items()}
        cats = {"Heatwave (heat-only)": masks["Heatwave"], "Drought (drought-only)": masks["Drought"],
                "Compound": masks["Compound"],
                "Normal": np.asarray(w["normal"], bool)}
        for cat, m in cats.items():
            for v in ("h", "d"):
                x = reg[v].values[m]
                rows.append({"window": wname, "category": cat, "variable": v, "n": int(m.sum()),
                             "mean": float(np.nanmean(x)), "min": float(np.nanmin(x)), "max": float(np.nanmax(x))})
    return pd.DataFrame(rows)


def b1_intensity(ctx, cfg: Cfg, choice: dict) -> tuple:
    rows = []
    for o in ctx["outcomes"]:
        w = ctx["windows"][o["window"]]
        reg = w["reg"]
        variants = {"": None, "_hd": {"h": reg["h"], "d": reg["d"]}}
        if o["window"] == "generation":
            variants["_d12"] = {"d12": reg["d12"]}
            variants["_hd_d12"] = {"h": reg["h"], "d": reg["d"], "d12": reg["d12"]}
        rec = {"Series": o["name"], "Group": o["group"], "Family": o["family"]}
        for suffix, extra in variants.items():
            t = fit(o, ctx, cfg, extra=extra, windows=False, boot=False)["table"].set_index("Term")
            if "Compound" not in t.index:
                continue
            rec["b_C" + suffix] = t.loc["Compound", "estimate"]
            rec["p_norm_shift_C" + suffix] = t.loc["Compound", "p_norm_shift"]
            rec["p_exact_shift_C" + suffix] = t.loc["Compound", "p_exact_shift"]
        for suffix in ("_hd", "_d12", "_hd_d12"):
            if "b_C" + suffix in rec:
                rec["ratio" + suffix] = rec["b_C" + suffix] / rec["b_C"] if rec["b_C"] else np.nan
        rec["unit"] = o["unit"]
        rows.append(rec)
    return pd.DataFrame(rows), _driver_summary(ctx)


def _models(reg, compound_mask, is_generation: bool) -> dict:
    h, d, hd = reg["h"], reg["d"], reg["hd"]
    m = {"M1": {"h": h, "d": d}, "M2": {"h": h, "d": d, "hd": hd}, "M3": {"h": h, "d": d, "hd": hd, "c": reg["c"]},
         "M4": {"h": h, "d": d, "hd": hd, "C": compound_mask}}
    if is_generation:
        m["M2w"] = {"h": h, "d": d, "hd": hd, "d12": reg["d12"]}
        m["M2w'"] = {"h": h, "d12": reg["d12"], "h_x_d12": h * reg["d12"]}
    return m


PARENT = {"M1": "M0", "M2": "M1", "M3": "M2", "M4": "M2", "M2w": "M2", "M2w'": "M0"}


def b1_continuous(ctx, cfg: Cfg, choice: dict) -> tuple:
    coef_rows, cmp_rows, marg_rows = [], [], []
    for o in ctx["outcomes"]:
        w = ctx["windows"][o["window"]]
        idx, y = w["index"], o["y"]
        comp = pd.Series(np.asarray(w["masks"]["Compound"], bool).astype(float), index=idx)
        models = _models(w["reg"], comp, o["window"] == "generation")
        Zb, _ = design_matrix(idx, None, trend=o["trend"])
        yv = y.values
        r2 = {"M0": ols_r2(yv, Zb)}
        cv = {"M0": loyo_cv_rmse(yv, lambda i: Zb, idx)}
        fam = o["family"]
        pt = (choice.get("families", {}).get(fam, {}) or {}).get("chosen")
        for mname, terms in models.items():
            out = fit_continuous_model(y, terms, o["trend"], n_perm=cfg.n_perm, n_boot=cfg.n_boot, seed=cfg.seed)
            t = out["table"]
            for _, r in t.iterrows():
                coef_rows.append({"Series": o["name"], "Group": o["group"], "Family": fam, "Model": mname,
                                  "Term": r["Term"], "estimate": r["estimate"],
                                  **{c: r[c] for c in INFERENCE_COLS},
                                  "p_chosen": r[pt] if pt else np.nan, "chosen_p_type": pt or ""})
            T = np.column_stack([v.values for v in terms.values()])
            X = np.column_stack([Zb, T])
            r2[mname] = out["r2"]
            cv[mname] = loyo_cv_rmse(yv, lambda i, X=X: X, idx)
            if mname == "M2":
                dr = out["boot"]["term_draws"] if out["boot"] else None
                names = list(terms)
                for lev in (0.0, D_LEVEL):
                    est = out["fit"]["beta"][-3 + names.index("h")] + lev * out["fit"]["beta"][-3 + names.index("hd")]
                    hc = hac_contrast(out["fit"], {"h": 1.0, "hd": lev})
                    lo = hi = np.nan
                    if dr is not None and len(dr) >= 20:
                        comb = dr[:, names.index("h")] + lev * dr[:, names.index("hd")]
                        lo, hi = np.percentile(comb, [2.5, 97.5])
                    marg_rows.append({"Series": o["name"], "Group": o["group"], "d_level": lev,
                                      "marginal_heat": est, "hac_se": hc["se"], "p_hac": hc["p"],
                                      "CI_boot_lo": lo, "CI_boot_hi": hi, "unit": o["unit"]})
        rows_models = ["M0"] + list(models)
        for mname in rows_models:
            rec = {"Series": o["name"], "Group": o["group"], "Model": mname, "R2": r2[mname],
                   "cv_rmse": cv[mname], "parent": PARENT.get(mname, ""), "dR2": np.nan, "dR2_p": np.nan}
            if mname != "M0":
                par = PARENT[mname]
                tpar = models[par] if par != "M0" else {}
                Zp = np.column_stack([Zb] + [v.values for v in tpar.values()])
                add_names = [k for k in models[mname] if k not in tpar]
                Tadd = np.column_stack([models[mname][k].values for k in add_names])
                if mname == "M2w'":
                    Tadd = np.column_stack([v.values for v in models[mname].values()])
                inc = incremental_r2_blocks(yv, Zp, Tadd, n_perm=cfg.n_r2, seed=cfg.seed)
                rec["dR2"], rec["dR2_p"] = inc["delta_r2"], inc["p_value"]
            cmp_rows.append(rec)
    return pd.DataFrame(coef_rows), pd.DataFrame(cmp_rows), pd.DataFrame(marg_rows)


EXPECTED_SIGN = {"Hydro share": -1, "Hydro GWh": -1, "Fossil share": 1, "Fossil GWh": 1, "Imbalance": 1}


def b1_decision(ctx, coef: pd.DataFrame, cmp_: pd.DataFrame, intens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for o in ctx["outcomes"]:
        n = o["name"]
        sign = EXPECTED_SIGN.get(n, 1 if o["group"] == "D" else np.nan)
        r = coef[(coef["Series"] == n) & (coef["Model"] == "M2") & (coef["Term"] == "hd")]
        i = intens[intens["Series"] == n].iloc[0]
        cv = cmp_[cmp_["Series"] == n].set_index("Model")["cv_rmse"]
        rec = {"Series": n, "Group": o["group"], "expected_sign_hd": sign}
        if len(r):
            r = r.iloc[0]
            ci_excl = bool(r["CI_boot_lo"] > 0 or r["CI_boot_hi"] < 0) if np.isfinite(r["CI_boot_lo"]) else False
            rec.update({"b_hd": r["estimate"], "p_chosen_hd": r["p_chosen"], "CI_boot_lo": r["CI_boot_lo"],
                        "CI_boot_hi": r["CI_boot_hi"], "chosen_p_type": r["chosen_p_type"]})
            rec["i_sign_p_ci"] = bool(np.isfinite(sign) and np.sign(r["estimate"]) == sign
                                      and r["p_chosen"] < 0.05 and ci_excl)
        rec["ii_cv_M3_or_M4_below_M2"] = bool(min(cv["M3"], cv["M4"]) < cv["M2"])
        rec["iii_ratio_hd"] = i.get("ratio_hd", np.nan)
        rec["iii_keeps_half"] = bool(i.get("ratio_hd", np.nan) >= 0.5)
        rec["compound_specific"] = bool(rec.get("i_sign_p_ci", False) and rec["ii_cv_M3_or_M4_below_M2"]
                                        and rec["iii_keeps_half"])
        if o["window"] == "generation":
            rec["ratio_hd_d12"] = i.get("ratio_hd_d12", np.nan)
            rec["dry_year_association"] = bool(i.get("ratio_hd_d12", np.nan) < 0.5)
        rows.append(rec)
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------------------------- B2
def _spec_extra(spec, ctx, o):
    reg = ctx["windows"]["generation"]["reg"]
    songs, nuc = ctx["extras"]["songs"], ctx["extras"]["nuclear_gwh"]
    return {"S0": (None, None), "S1": (None, "year_fe"), "S2": ({"songs": songs}, None),
            "S3": ({"nuclear_gwh": nuc}, None), "S4": ({"songs": songs, "d12": reg["d12"]}, None)}[spec]


def b2_specifications(ctx, cfg: Cfg, choice: dict) -> tuple:
    rows = []
    for o in _series(ctx, groups=["G-share", "G-energy"]):
        for spec in ("S0", "S1", "S2", "S3", "S4"):
            if spec == "S3" and o["col"] == "Nuclear":
                continue
            extra, trend = _spec_extra(spec, ctx, o)
            t = fit(o, ctx, cfg, extra=extra, trend=trend)["table"]
            for _, r in t.iterrows():
                rows.append({"Series": o["name"], "Group": o["group"], "Family": "generation", "Spec": spec,
                             "Term": r["Term"], "N": r["N"], "estimate": r["estimate"], "unit": o["unit"],
                             "Delta_pct": r["Delta_pct"], "CI_shift_lo": r["ci_lo"], "CI_shift_hi": r["ci_hi"],
                             **{c: r[c] for c in INFERENCE_COLS}})
    tab = pd.DataFrame(rows)
    tab["p_chosen"] = chosen_p(tab["Family"], tab, choice)
    return tab, {"boot_draws_dropped": int(tab["n_boot_dropped"].max())}


def b2_leave_one_episode_out(ctx, cfg: Cfg) -> tuple:
    w = ctx["windows"]["generation"]
    m = pd.Series(np.asarray(w["masks"]["Compound"], bool), index=w["index"])
    eps = compound_episodes(m)
    wy = water_year_labels(w["index"])
    yr = w["index"].year.values
    loeo, jack = [], []
    for o in _series(ctx, groups=["G-share", "G-energy"]):
        full = fit(o, ctx, cfg, windows=False, boot=False)["table"].set_index("Term").loc["Compound"]
        for k, ((a, b), (_, e)) in enumerate(zip(episode_water_years(eps), eps.iterrows())):
            ex = (wy >= a) & (wy <= b)
            t = fit(o, ctx, cfg, exclude=ex, windows=False, boot=False)["table"].set_index("Term")
            if "Compound" not in t.index:
                loeo.append({"Series": o["name"], "episode": f"{e['start']:%Y-%m}..{e['end']:%Y-%m}",
                             "water_years_dropped": f"{a}-{b}", "N_compound_left": 0})
                continue
            c = t.loc["Compound"]
            loeo.append({"Series": o["name"], "episode": f"{e['start']:%Y-%m}..{e['end']:%Y-%m}",
                         "water_years_dropped": f"{a}-{b}", "b_C": c["estimate"], "CI_lo": c["ci_lo"],
                         "CI_hi": c["ci_hi"], "p_norm_shift": c["p_norm_shift"], "p_exact_shift": c["p_exact_shift"],
                         "N_compound_left": int(c["N"]), "same_sign_as_full": bool(np.sign(c["estimate"]) == np.sign(full["estimate"]))})
        vals = {}
        for y_ in np.unique(yr):
            t = fit(o, ctx, cfg, exclude=(yr == y_), windows=False, boot=False)["table"].set_index("Term")
            if "Compound" in t.index:
                vals[int(y_)] = float(t.loc["Compound", "estimate"])
        s = pd.Series(vals)
        jack.append({"Series": o["name"], "b_C_full": full["estimate"], "n_fits": int(len(s)), "min": s.min(),
                     "year_at_min": int(s.idxmin()), "max": s.max(), "year_at_max": int(s.idxmax()),
                     "n_same_sign": int((np.sign(s) == np.sign(full["estimate"])).sum())})
    return eps, pd.DataFrame(loeo), pd.DataFrame(jack)


def b2_substitution(ctx, cfg: Cfg, spec_table: pd.DataFrame) -> tuple:
    w = ctx["windows"]["generation"]
    E = ctx["energy"]
    rows, dec = [], []
    for spec in ("S0", "S2"):
        extra, _ = _spec_extra(spec, ctx, None)
        s = substitution_accounting(E, w["masks"], w["normal"], "quadratic", extra=extra, n_boot=cfg.n_boot,
                                    seed=cfg.seed)
        rows.append({"Spec": spec, "Event": "Compound", **s})
        for src in ("Hydro", "Fossil"):
            sh = spec_table[(spec_table["Series"] == f"{src} share") & (spec_table["Spec"] == spec)
                            & (spec_table["Term"] == "Compound")].iloc[0]
            dec.append({"Spec": spec, "Source": src, **share_decomposition(
                sh["estimate"], s[f"b_{src}"], s["b_Total"], s[f"baseline_{src}"], s["baseline_Total"])})
    return pd.DataFrame(rows), pd.DataFrame(dec)


def b2_decision(spec_table: pd.DataFrame, loeo: pd.DataFrame, choice_p: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for name in ("Fossil share", "Fossil GWh", "Hydro share", "Hydro GWh"):
        s = spec_table[(spec_table["Series"] == name) & (spec_table["Term"] == "Compound")].set_index("Spec")
        full = s.loc["S0", "estimate"]
        l = loeo[loeo["Series"] == name]
        sign_ok = bool((np.sign(s["estimate"]) == np.sign(full)).all())
        loeo_ok = bool(l["same_sign_as_full"].fillna(False).all()) if len(l) else False
        p1, p2 = s.loc["S1", "p_chosen"], s.loc["S2", "p_chosen"]
        smallest = s["estimate"].abs().idxmin()
        rows.append({"Series": name, "sign_kept_S0_S4": sign_ok, "sign_kept_all_LOEO": loeo_ok,
                     "p_chosen_S1": p1, "p_chosen_S2": p2,
                     "robust_to_portfolio_change": bool(sign_ok and loeo_ok and p1 < 0.05 and p2 < 0.05),
                     "smallest_effect_spec": smallest, "smallest_effect": s.loc[smallest, "estimate"],
                     "unit": s.loc[smallest, "unit"], "S0_effect": full})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- checks and add-ons
FIGURE_CSV = {"G-share": ("figure_02", "figure_02_composite.csv", "Source", {"Mode": "adjusted_share"}),
              "D": ("figure_04", "figure_04_event_composites.csv", "Region", {}),
              "I": ("figure_06", "figure_06_event_metrics.csv", None, {})}


def reproduction_checks(ctx, figures_dir, b1: pd.DataFrame | None, b2: pd.DataFrame | None) -> pd.DataFrame:
    """Max |difference| of B1a (and the S0 rows of B2) against the published figure CSVs."""
    from pathlib import Path

    col_of = {o["name"]: o["col"] for o in ctx["outcomes"]}
    rows = []
    for group, (fdir, fname, key, sel) in FIGURE_CSV.items():
        path = Path(figures_dir) / fdir / fname
        if not path.exists():
            rows.append({"table": group, "source": fname, "status": "figure CSV missing"})
            continue
        ref = pd.read_csv(path)
        for k, v in sel.items():
            ref = ref[ref[k] == v]
        for label, tab, term_col, cols in (
                ("B1a", b1, "Event", {"Delta_native": "Delta_native", "Delta_pct": "Delta_pct", "Delta_z": "Delta_z",
                                      "p_norm_shift": "p_value", "p_exact_shift": "p_exact", "q_norm_shift": "q_value"}),
                ("B2-S0", b2, "Term", {"estimate": "Delta_native", "Delta_pct": "Delta_pct",
                                       "p_norm_shift": "p_value", "p_exact_shift": "p_exact"})):
            if tab is None or (label == "B2-S0" and group != "G-share"):
                continue
            t = tab[tab["Group"] == group]
            if label == "B2-S0":
                t = t[t["Spec"] == "S0"]
            diffs = {}
            for _, r in t.iterrows():
                m = ref[(ref["Event"] == r[term_col])]
                if key:
                    m = m[m[key] == col_of[r["Series"]]]
                if len(m) != 1:
                    continue
                for ours, theirs in cols.items():
                    d = abs(float(r[ours]) - float(m.iloc[0][theirs]))
                    diffs[theirs] = max(diffs.get(theirs, 0.0), d if np.isfinite(d) else 0.0)
            rows.append({"table": f"{label} {group}", "source": fname, "n_rows": int(len(t)),
                         **{f"max_abs_diff_{k}": v for k, v in diffs.items()}, "status": "ok"})
    return pd.DataFrame(rows)


def addons(ctx, root) -> dict:
    """A5 series spans and the provenance of the simulation claim in INFERENCE_NOTE."""
    import subprocess

    from componergy.analysis.index_diagnostics import standardized_precip_index, temperature_dependence

    drv = ctx["drivers"]

    def span(s):
        s = s.dropna()
        return {"first": str(s.index[0].date()), "last": str(s.index[-1].date()), "count": int(len(s))}

    raw = -drv["scdhi3"]
    smooth = raw.rolling(3, center=True, min_periods=1).mean()
    spi = standardized_precip_index(drv["prcp_raw"], scale=3) if "prcp_raw" in drv else None
    a5 = {"-SCDHI smoothed (3-mo centred MA, min_periods=1)": span(smooth), "-SCDHI unsmoothed": span(raw)}
    if spi is not None:
        a5["SPI-3 (SPI check series)"] = span(spi)
        a5["months shared by STI, SAPEI-3, SPI-3 (SPI check n_months)"] = temperature_dependence(
            drv["sti"], drv["sapei3"], spi).get("n_months")
    a5["explanation"] = {
        "expected": "unsmoothed starts Mar 1951 (906); min_periods=1 smoothing back-fills Feb 1951 (907)",
        "unsmoothed_first_is_1951-03_and_count_906": a5["-SCDHI unsmoothed"]["first"] == "1951-03-01"
        and a5["-SCDHI unsmoothed"]["count"] == 906,
        "smoothed_first_is_1951-02_and_count_907": a5["-SCDHI smoothed (3-mo centred MA, min_periods=1)"]["first"]
        == "1951-02-01" and a5["-SCDHI smoothed (3-mo centred MA, min_periods=1)"]["count"] == 907}

    def git(*args):
        r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return r.stdout.strip() or r.stderr.strip()

    prov = {"git log -S '4-8%' --all --oneline": git("log", "-S", "4-8%", "--all", "--oneline"),
            "git log -S '4\u20138%' (en dash) --all --oneline": git("log", "-S", "4\u20138%", "--all", "--oneline"),
            "git log -S 'In simulation on series with no effect' --all --oneline":
                git("log", "-S", "In simulation on series with no effect", "--all", "--oneline")}
    first = [l.split()[0] for l in prov["git log -S 'In simulation on series with no effect' --all --oneline"].splitlines() if l]
    if first:
        prov["files touched by the oldest such commit"] = git("show", "--stat", "--format=%h %an %ad %s", first[-1])
    import os
    prov["exploratory/ directory present"] = os.path.isdir(os.path.join(root, "exploratory"))
    prov["simulation scripts found (grep 'no effect|null simulation' in scripts/, src/, notebooks/)"] = git(
        "grep", "-n", "-i", "-E", "series with no effect|null simulation|nominal 5%", "--", "scripts", "src", "notebooks")
    return {"A5": a5, "INFERENCE_NOTE_provenance": prov}
