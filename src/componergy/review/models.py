# src/componergy/review/models.py
"""
Event and continuous regression models with the full set of review inference types.

Both model families are one problem: an outcome y, a nuisance design Z (constant, month effects, trend and
optional controls that are never shifted) and a term block T (event dummies, or continuous drivers) whose
coefficients are tested. `analyse` returns, for the same point estimate, the circular-shift p (normal and
exact), the season-preserving shift p, the water-year block-permutation p, HAC(12) and a pairs-bootstrap CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from componergy.analysis.events import exclusive_event_masks
from componergy.analysis.lag_scan import block_permute
from componergy.analysis.regression import (
    DEFAULT_HAC_LAGS, DEFAULT_MIN_EVENT, design_matrix, event_effects, hac_contrast, ols_hac,
    randomization_inference, shift_null_coefficients,
)
from componergy.review.inference import (
    SEED, pairs_block_bootstrap, season_preserving_shifts, shift_null_generic, water_year_labels,
    whole_year_window, year_block_permutation_null,
)

AMPLIFICATION = {"Compound": 1.0, "Heatwave": -1.0, "Drought": -1.0}
P_TYPES = ("p_norm_shift", "p_exact_shift", "p_exact_season", "p_yearperm", "p_hac")


@dataclass
class TermProblem:
    """One regression: outcome y, term block T (N x k, all-zero rows allowed outside `sample`), nuisance controls."""
    index: pd.DatetimeIndex
    y: np.ndarray
    T: np.ndarray
    names: list
    sample: np.ndarray
    trend: str
    controls: np.ndarray | None = None
    control_names: list = field(default_factory=list)
    contrasts: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    binary: bool = False


def _as_bool(s, index) -> np.ndarray:
    if isinstance(s, np.ndarray):  # positional: already aligned with `index`
        return s.astype(bool)
    return pd.Series(s).reindex(index).fillna(False).astype(bool).values


def build_event_problem(y, masks: dict, normal, trend: str, extra: dict | None = None, exclude=None,
                        min_event: int = DEFAULT_MIN_EVENT, exclusive: bool = True) -> TermProblem:
    """Event dummies exactly as `regression.event_effects` forms them, plus optional unshifted controls."""
    y = pd.Series(y).astype(float)
    idx = pd.DatetimeIndex(y.index)
    m = {k: pd.Series(_as_bool(v, idx), index=idx) for k, v in masks.items()}
    if exclusive:
        m = exclusive_event_masks(m)
    in_sample = _as_bool(normal, idx).copy()
    for v in m.values():
        in_sample |= v.values
    in_sample &= y.notna().values
    controls = None
    if extra:
        controls = np.column_stack([pd.Series(v).reindex(idx).astype(float).values for v in extra.values()])
        in_sample &= np.isfinite(controls).all(axis=1)
    if exclude is not None:
        in_sample &= ~_as_bool(exclude, idx)
    counts = {k: int(v.values[in_sample].sum()) for k, v in m.items()}
    names = [k for k in m if counts[k] >= min_event]
    T = np.column_stack([m[k].values.astype(float) * in_sample for k in names]) if names else np.empty((len(idx), 0))
    contrasts = {"A": AMPLIFICATION} if all(k in names for k in AMPLIFICATION) else {}
    return TermProblem(idx, y.values, T, names, in_sample, trend, controls, list(extra or {}), contrasts,
                       counts, binary=True)


def build_continuous_problem(y, terms: dict, trend: str, controls: dict | None = None) -> TermProblem:
    """All valid months: y and every term finite."""
    y = pd.Series(y).astype(float)
    idx = pd.DatetimeIndex(y.index)
    T = np.column_stack([pd.Series(v).reindex(idx).astype(float).values for v in terms.values()]) \
        if terms else np.empty((len(idx), 0))
    ctrl = np.column_stack([pd.Series(v).reindex(idx).astype(float).values for v in controls.values()]) \
        if controls else None
    sample = y.notna().values & np.isfinite(T).all(axis=1)
    if ctrl is not None:
        sample &= np.isfinite(ctrl).all(axis=1)
    T = np.where(sample[:, None], T, 0.0)
    return TermProblem(idx, y.values, T, list(terms), sample, trend, ctrl, list(controls or {}), {},
                       {k: int(sample.sum()) for k in terms})


def _design(prob: TermProblem, rows) -> tuple:
    Z, names = design_matrix(prob.index[rows], None, trend=prob.trend)
    if prob.controls is not None:
        Z = np.column_stack([Z, prob.controls[rows]])
        names = names + list(prob.control_names)
    return Z, names


def _window(prob: TermProblem, kind: str) -> dict:
    """The problem restricted to a whole-year window; non-sample months keep their calendar slot (`keep`)."""
    win = whole_year_window(prob.index, kind)
    wm = np.asarray(prob.index.isin(win))
    keep = prob.sample[wm]
    Zk, _ = _design(prob, prob.sample & wm)
    Zfull = np.zeros((int(wm.sum()), Zk.shape[1]))
    Zfull[keep] = Zk
    T = prob.T[wm]
    y = np.where(keep, prob.y[wm], 0.0)
    valid = np.abs(T[keep]).sum(axis=0) > 0 if T.shape[1] else np.zeros(0, bool)
    est = np.full(T.shape[1], np.nan)
    if valid.any():
        X = np.column_stack([Zk, T[keep][:, valid]])
        est[valid] = np.linalg.lstsq(X, y[keep], rcond=None)[0][-int(valid.sum()):]
    return {"index": prob.index[wm], "y": y, "Z": Zfull, "Zk": Zk, "T": T, "keep": keep, "valid": valid, "est": est}


def _apply(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """A (S, k) -> (S, m) combination by the rows of W, ignoring (NaN) columns that carry zero weight."""
    out = np.empty((len(A), len(W)))
    for i, w in enumerate(W):
        nz = np.flatnonzero(w)
        out[:, i] = A[:, nz] @ w[nz]
    return out


def _expand(null: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full((len(null), len(valid)), np.nan)
    out[:, valid] = null
    return out


def _contrast_matrix(prob: TermProblem):
    names, rows, labels = prob.names, [], []
    for j, nm in enumerate(names):
        w = np.zeros(len(names))
        w[j] = 1.0
        rows.append(w)
        labels.append(nm)
    for cname, wd in prob.contrasts.items():
        w = np.zeros(len(names))
        for k, v in wd.items():
            w[names.index(k)] = v
        rows.append(w)
        labels.append(cname)
    return np.array(rows), labels


def analyse(prob: TermProblem, n_perm: int = 1999, n_boot: int = 2000, seed: int = SEED, boot: bool = True,
            hac_lags: int = DEFAULT_HAC_LAGS, windows: bool = True) -> dict:
    """Every inference type for every term (and contrast) of `prob`; see the module docstring."""
    k = len(prob.names)
    W, labels = _contrast_matrix(prob)
    rows = prob.sample
    Z, zn = _design(prob, rows)
    yk, Tk = prob.y[rows], prob.T[rows]
    X = np.column_stack([Z, Tk])
    fit = ols_hac(yk, X, zn + prob.names, lags=hac_lags)
    beta = fit["beta"][-k:]
    null = shift_null_coefficients(yk, Z, Tk)
    est = W @ beta
    null_m = _apply(null, W)

    nan = np.full(len(labels), np.nan)
    cols = {c: nan.copy() for c in ("est_season", "p_norm_season", "p_exact_season", "n_season", "est_water",
                                    "p_norm_yearperm", "p_yearperm", "n_perm", "CI_boot_lo", "CI_boot_hi",
                                    "n_boot_ok", "n_boot_dropped")}
    boot_out = None
    if windows and k:
        cal = _window(prob, "calendar")
        wat = _window(prob, "water")
        for win, tag in ((cal, "season"), (wat, "water")):
            cols["est_" + tag][:] = _apply(win["est"][None, :], W)[0]
        if cal["valid"].any():
            nul = _expand(shift_null_generic(cal["y"], cal["Z"], cal["T"][:, cal["valid"]],
                                             season_preserving_shifts(len(cal["y"])), keep=cal["keep"]),
                          cal["valid"])
            nul = _apply(nul, W)
            for i in range(len(labels)):
                p = _p(cols["est_season"][i], nul[:, i])
                cols["p_norm_season"][i], cols["p_exact_season"][i], cols["n_season"][i] = p
        if wat["valid"].any():
            if n_perm > 0:
                nul = _expand(year_block_permutation_null(wat["y"], wat["Z"], wat["T"][:, wat["valid"]],
                                                          wat["index"], "water", n_perm, seed, keep=wat["keep"]),
                              wat["valid"])
                nul = _apply(nul, W)
                for i in range(len(labels)):
                    p = _p(cols["est_water"][i], nul[:, i])
                    cols["p_norm_yearperm"][i], cols["p_yearperm"][i], cols["n_perm"][i] = p
            if boot and n_boot > 0:
                v = wat["valid"]
                Xb = np.column_stack([wat["Zk"], wat["T"][wat["keep"]][:, v]])
                nz = int(v.sum())
                b = pairs_block_bootstrap(wat["y"][wat["keep"]], Xb, wat["index"][wat["keep"]],
                                          list(range(Xb.shape[1] - nz, Xb.shape[1])), n_boot=n_boot, seed=seed)
                dr = np.full((len(b["draws"]), len(v)), np.nan)
                dr[:, v] = b["draws"]
                dm = _apply(dr, W)
                if len(dm) >= 20:
                    cols["CI_boot_lo"][:], cols["CI_boot_hi"][:] = np.percentile(dm, [2.5, 97.5], axis=0)
                cols["n_boot_ok"][:], cols["n_boot_dropped"][:] = b["n_ok"], b["n_dropped"]
                boot_out = {"draws": dm, "labels": labels, "term_draws": dr, "n_dropped": b["n_dropped"],
                            "n_ok": b["n_ok"]}

    base = {}
    if prob.binary:
        X0 = X.copy()
        X0[:, -k:] = 0.0
        pred0 = X0 @ fit["beta"]
        for j, nm in enumerate(prob.names):
            base[nm] = float(np.mean(pred0[Tk[:, j] > 0]))

    recs = []
    for i, lab in enumerate(labels):
        ri = randomization_inference(float(est[i]), null_m[:, i])
        hc = hac_contrast(fit, {prob.names[j]: W[i, j] for j in np.flatnonzero(W[i])})
        rec = {"Term": lab, "N": prob.counts.get(lab, np.nan), "estimate": float(est[i]), "se": ri["se"],
               "ci_lo": ri["ci_lo"], "ci_hi": ri["ci_hi"], "p_norm_shift": ri["p_value"],
               "p_exact_shift": ri["p_exact"], "p_hac": hc["p"], "n_shifts": ri["n_shifts"],
               "Baseline_mean": base.get(lab, np.nan)}
        rec["Delta_pct"] = 100.0 * est[i] / rec["Baseline_mean"] if rec["Baseline_mean"] else np.nan
        rec.update({c: v[i] for c, v in cols.items()})
        recs.append(rec)
    return {"table": pd.DataFrame(recs), "fit": fit, "null": null, "boot": boot_out, "labels": labels,
            "W": W, "Z": Z, "zn": zn, "n_used": int(rows.sum())}


def _p(est, null):
    r = randomization_inference(est, null)
    return r["p_value"], r["p_exact"], r["n_shifts"]


def fit_event_model(y, masks: dict, normal, trend: str, extra: dict | None = None, **kw) -> dict:
    """
    Event effects with every inference type. With `extra=None` the `effects` table is
    `regression.event_effects` verbatim (new inference columns appended); with `extra` the continuous
    columns enter the nuisance design (controls, not shifted) and the same columns are rebuilt.
    """
    prob = build_event_problem(y, masks, normal, trend, extra=extra)
    out = analyse(prob, **kw)
    t = out["table"].set_index("Term")
    if extra is None:
        ref = event_effects(y, masks, normal, trend=trend)
        eff = ref["effects"].copy()
        out["event_effects"] = ref
    else:
        eff = pd.DataFrame({"Event": list(masks)})
        eff["N_event"] = [prob.counts.get(e, 0) for e in eff["Event"]]
        for src, dst in (("estimate", "Delta_native"), ("se", "SE_native"), ("ci_lo", "CI_lo_native"),
                         ("ci_hi", "CI_hi_native"), ("p_norm_shift", "p_value"), ("p_exact_shift", "p_exact"),
                         ("p_hac", "p_hac"), ("Baseline_mean", "Baseline_mean"), ("Delta_pct", "Delta_pct")):
            eff[dst] = [t[src].get(e, np.nan) for e in eff["Event"]]
    for c in ("est_season", "p_exact_season", "est_water", "p_yearperm", "CI_boot_lo", "CI_boot_hi"):
        eff[c] = [t[c].get(e, np.nan) for e in eff["Event"]]
    out["effects"] = eff
    out["problem"] = prob
    return out


def fit_continuous_model(y, terms: dict, trend: str, sample: str = "valid", controls: dict | None = None,
                         **kw) -> dict:
    """OLS y ~ month FE + trend + terms on all valid months; the term block is shifted/permuted jointly."""
    if sample != "valid":
        raise ValueError("only sample='valid' (every month with finite y and terms) is implemented")
    prob = build_continuous_problem(y, terms, trend, controls)
    out = analyse(prob, **kw)
    out["problem"] = prob
    out["r2"] = out["fit"]["r2"]
    return out


def ols_r2(y, Z) -> float:
    y = np.asarray(y, dtype=float)
    beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    r = y - Z @ beta
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(r @ r) / sst if sst > 0 else np.nan


def loyo_cv_rmse(y, design_fn, index) -> float:
    """Leave-one-water-year-out cross-validated RMSE of the OLS model whose design is `design_fn(index)`."""
    idx = pd.DatetimeIndex(index)
    y = np.asarray(y, dtype=float)
    X = np.asarray(design_fn(idx), dtype=float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    labels = water_year_labels(idx)
    sse, n = 0.0, 0
    for wy in np.unique(labels[ok]):
        test = ok & (labels == wy)
        train = ok & (labels != wy)
        beta = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
        r = y[test] - X[test] @ beta
        sse += float(r @ r)
        n += int(test.sum())
    return float(np.sqrt(sse / n)) if n else np.nan


def incremental_r2_blocks(y, Z_prev, T_add, n_perm: int = 1999, block: int = 24, seed: int = SEED) -> dict:
    """
    Increase in R^2 from adding the columns `T_add` to a model with nuisance/previous design `Z_prev`, against a
    null that permutes the rows of T_add jointly in `block`-month blocks. Generalises
    `relative_importance.incremental_r2_block_permutation` (one added series, intercept-only base) to a full
    base design and several added columns; uses the same `block_permute` and p-value convention.
    """
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z_prev, dtype=float)
    T = np.asarray(T_add, dtype=float)
    T = T[:, None] if T.ndim == 1 else T
    n = len(y)
    M = np.eye(n) - Z @ np.linalg.pinv(Z)
    yt = M @ y
    sst = float(np.sum((y - y.mean()) ** 2))

    def gain(Tm):
        Tr = M @ Tm
        c = np.linalg.lstsq(Tr, yt, rcond=None)[0]
        return float(yt @ (Tr @ c)) / sst

    obs = gain(T)
    rng = np.random.default_rng(seed)
    pos = np.arange(n)
    null = np.array([gain(T[block_permute(pos, block, rng)]) for _ in range(int(n_perm))])
    return {"delta_r2": obs, "p_value": float((1 + np.sum(null >= obs)) / (len(null) + 1)), "n_perm": int(len(null))}


def substitution_ratios(b_fossil: float, b_hydro: float, b_total: float) -> dict:
    """R_fossil = b(Fossil)/(-b(Hydro)); R_unreplaced = b(Total)/b(Hydro): share of the hydro shortfall not replaced."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"R_fossil": b_fossil / -b_hydro if b_hydro else np.nan,
                "R_unreplaced": b_total / b_hydro if b_hydro else np.nan}


def substitution_accounting(Y: pd.DataFrame, masks: dict, normal, trend: str, extra: dict | None = None,
                            event: str = "Compound", n_boot: int = 2000, seed: int = SEED) -> dict:
    """
    Fossil/Hydro/Total GWh event effects (columns of Y), the two substitution ratios and their pairs-bootstrap
    percentile CI; both coefficients of a ratio come from the same water-year block draw.
    """
    probs = {c: build_event_problem(Y[c], masks, normal, trend, extra=extra) for c in ("Fossil", "Hydro", "Total")}
    ref = probs["Hydro"]
    if event not in ref.names:
        return {}
    j = ref.names.index(event)
    b = {}
    for c, p in probs.items():
        Z, _ = _design(p, p.sample)
        X = np.column_stack([Z, p.T[p.sample]])
        b[c] = float(np.linalg.lstsq(X, p.y[p.sample], rcond=None)[0][-len(p.names) + j])
    out = {"b_Fossil": b["Fossil"], "b_Hydro": b["Hydro"], "b_Total": b["Total"],
           **substitution_ratios(b["Fossil"], b["Hydro"], b["Total"])}
    wat = _window(ref, "water")
    keep = wat["keep"]
    Yw = np.column_stack([np.where(keep, Y[c].reindex(ref.index).values[np.asarray(ref.index.isin(wat["index"]))], 0)
                          for c in ("Fossil", "Hydro", "Total")])
    Xb = np.column_stack([wat["Zk"], wat["T"][keep]])
    bb = pairs_block_bootstrap(Yw[keep], Xb, wat["index"][keep], [Xb.shape[1] - len(ref.names) + j],
                               n_boot=n_boot, seed=seed)
    d = bb["draws"][:, 0, :]  # (B, 3): Fossil, Hydro, Total
    rf = d[:, 0] / -d[:, 1]
    ru = d[:, 2] / d[:, 1]
    q = lambda a: tuple(np.percentile(a, [2.5, 97.5])) if len(a) >= 20 else (np.nan, np.nan)
    out.update({"R_fossil_lo": q(rf)[0], "R_fossil_hi": q(rf)[1], "R_unreplaced_lo": q(ru)[0],
                "R_unreplaced_hi": q(ru)[1], "n_boot_ok": bb["n_ok"], "n_boot_dropped": bb["n_dropped"],
                "baseline_Fossil": float(_baseline(probs["Fossil"], j)),
                "baseline_Hydro": float(_baseline(probs["Hydro"], j)),
                "baseline_Total": float(_baseline(probs["Total"], j))})
    return out


def _baseline(p: TermProblem, j: int) -> float:
    """Mean no-event prediction over the months of term j."""
    Z, _ = _design(p, p.sample)
    X = np.column_stack([Z, p.T[p.sample]])
    beta = np.linalg.lstsq(X, p.y[p.sample], rcond=None)[0]
    X0 = X.copy()
    X0[:, -len(p.names):] = 0.0
    return float(np.mean((X0 @ beta)[p.T[p.sample][:, j] > 0]))


def share_decomposition(b_share_pp: float, b_x: float, b_total: float, base_x: float, base_total: float) -> dict:
    """
    First-order split of a share change (percentage points) into the numerator and the total:
    d(100 X/T) ~= 100 b_x / T0  -  100 X0 b_total / T0^2. `residual` is what the linearisation leaves.
    """
    num = 100.0 * b_x / base_total
    den = -100.0 * base_x * b_total / base_total ** 2
    return {"share_effect_pp": b_share_pp, "from_numerator_pp": num, "from_total_pp": den,
            "residual_pp": b_share_pp - num - den}
