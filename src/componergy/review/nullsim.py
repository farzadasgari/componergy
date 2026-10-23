# src/componergy/review/nullsim.py
"""
Null simulations for p-value calibration (B3c).

In both generators the outcome is the REAL series; only the climate, hence the event calendar, is replaced by
one that is independent of electricity. The true effect is zero by construction while the real seasonality,
trend, autocorrelation and portfolio breaks of the outcome are kept.

  N1  historical placebo climate: the real statewide indices of whole windows ending before the electricity
      record (<= Dec 2000), relabelled to the outcome calendar (month-of-year preserved); thresholds stay the
      real full-record ones.
  N2  parametric climate: per-calendar-month mean/SD and a linear trend removed, a VAR(p) (BIC) fitted to the
      standardised anomalies, then simulated, re-standardised and classified by the real pipeline.

Workers receive numpy arrays only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from componergy.analysis.events import build_event_masks
from componergy.review.models import P_TYPES, analyse, build_continuous_problem, build_event_problem

SEED = 42
TERMS = ("Compound", "Heatwave", "Drought", "A", "hd")
SMOOTH = 3
ALPHA = 0.05


def to_ns(index) -> np.ndarray:
    """Month timestamps as int64 nanoseconds, the form handed to workers."""
    return np.asarray(pd.DatetimeIndex(index).as_unit("ns").asi8)


def wilson_interval(k: float, n: float, z: float = 1.959964) -> tuple:
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(centre - half), float(centre + half))


def zscore_array(x: np.ndarray) -> np.ndarray:
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 0 else x * 0.0


def smooth_array(x: np.ndarray, months: int = SMOOTH) -> np.ndarray:
    return pd.Series(x).rolling(months, center=True, min_periods=1).mean().values


# --------------------------------------------------------------------------------------------- N2 simulator
def _month_design(months: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.column_stack([(months[:, None] == np.arange(12)[None, :]).astype(float), t])


def fit_var_simulator(raw: np.ndarray, months: np.ndarray, p_max: int = 6) -> dict:
    """
    Per-month mean + linear trend (OLS), per-month SD, then a VAR(p) for the standardised anomalies, p by BIC.

    raw: (T, 3) statewide [STI, SAPEI3, SCDHI3] (NaN where undefined); months: 0..11 per row.
    """
    T, k = raw.shape
    t = np.arange(T) / 120.0
    D = _month_design(months, t)
    ok = np.isfinite(raw).all(axis=1)
    coef = np.linalg.lstsq(D[ok], raw[ok], rcond=None)[0]  # (13, 3): month means then trend per decade
    resid = np.where(ok[:, None], raw - D @ coef, np.nan)
    sd = np.array([[np.sqrt(np.nanmean(resid[(months == m)][:, j] ** 2)) for j in range(k)] for m in range(12)])
    a = resid / sd[months]
    a = a[ok]  # contiguous valid block
    best = None
    n_common = len(a) - p_max
    for p in range(1, p_max + 1):
        Y = a[p_max:]
        X = np.column_stack([a[p_max - j:len(a) - j] for j in range(1, p + 1)])
        B = np.linalg.lstsq(X, Y, rcond=None)[0]
        E = Y - X @ B
        S = E.T @ E / n_common
        bic = np.linalg.slogdet(S)[1] + p * k * k * np.log(n_common) / n_common
        if best is None or bic < best[0]:
            best = (bic, p)
    p = best[1]
    Y = a[p:]
    X = np.column_stack([a[p - j:len(a) - j] for j in range(1, p + 1)])
    B = np.linalg.lstsq(X, Y, rcond=None)[0]  # (p*k, k)
    E = Y - X @ B
    Sigma = E.T @ E / (len(Y) - p * k)
    return {"coef": coef, "sd": sd, "B": B, "p": int(p), "Sigma": Sigma, "T": int(T), "months": months,
            "bic_p": int(p)}


def simulate_raw(params: dict, rng, burn: int = 120) -> np.ndarray:
    """One simulated (T, 3) raw statewide series on the calendar of the fit."""
    T, p, B = params["T"], params["p"], params["B"]
    k = B.shape[1]
    L = np.linalg.cholesky(params["Sigma"])
    n = T + burn
    a = np.zeros((n + p, k))
    eps = rng.standard_normal((n, k)) @ L.T
    for i in range(n):
        lagged = np.concatenate([a[p + i - j] for j in range(1, p + 1)])
        a[p + i] = lagged @ B + eps[i]
    a = a[p + burn:]
    months = params["months"]
    D = _month_design(months, np.arange(T) / 120.0)
    return D @ params["coef"] + params["sd"][months] * a


# ------------------------------------------------------------------------------------------ one draw
def run_lengths(mask: np.ndarray) -> float:
    m = np.asarray(mask, bool)
    starts = m & ~np.concatenate([[False], m[:-1]])
    return float(m.sum() / starts.sum()) if starts.sum() else np.nan


def evaluate_draw(sti: np.ndarray, sapei: np.ndarray, scdhi: np.ndarray, dates_ns: np.ndarray, win: dict,
                  n_perm: int, seed: int) -> dict:
    """
    Classify events from the climate series (z-scored, unsmoothed) on the window positions `win["pos"]`, relabel
    them to the outcome calendar, fit the primary event model and the continuous model for every outcome
    column, and return the p-values [series, term, p-type] plus event statistics.
    """
    dates = pd.DatetimeIndex(dates_ns)
    ser = {k: pd.Series(v, index=dates) for k, v in (("sti", sti), ("sapei", sapei), ("scdhi", scdhi))}
    win_idx = dates[win["pos"]]
    out_idx = pd.DatetimeIndex(win["index_ns"])
    masks, normal, meta, flags = build_event_masks(ser["sti"], ser["sapei"], ser["scdhi"], window_index=win_idx,
                                                   baseline="full_record", return_flags=True)
    relabel = lambda s: pd.Series(np.asarray(s.values, bool), index=out_idx)  # noqa: E731
    masks = {k: relabel(v) for k, v in masks.items()}
    normal = relabel(normal)
    h = smooth_array(sti)[win["pos"]]
    d = -smooth_array(sapei)[win["pos"]]
    terms = {"h": pd.Series(h, index=out_idx), "d": pd.Series(d, index=out_idx), "hd": pd.Series(h * d, index=out_idx)}

    names, trends, Y = win["names"], win["trends"], win["Y"]
    P = np.full((len(names), len(TERMS), len(P_TYPES)), np.nan)
    for j, (nm, trend) in enumerate(zip(names, trends)):
        y = pd.Series(Y[:, j], index=out_idx)
        ev = analyse(build_event_problem(y, masks, normal, trend), n_perm=n_perm, boot=False,
                     seed=seed + j)["table"].set_index("Term")
        co = analyse(build_continuous_problem(y, terms, trend), n_perm=n_perm, boot=False,
                     seed=seed + j)["table"].set_index("Term")
        for ti, term in enumerate(TERMS):
            src = co if term == "hd" else ev
            if term in src.index:
                P[j, ti] = src.loc[term, list(P_TYPES)].values.astype(float)
    comp = flags["compound"].values
    stats_ = {"compound_months": float(comp.sum()), "mean_compound_run": run_lengths(comp),
              "compound_heat_share": float((comp & flags["heat"].values).sum() / max(comp.sum(), 1)),
              "compound_drought_share": float((comp & flags["drought"].values).sum() / max(comp.sum(), 1)),
              "heat_months": float(flags["heat"].sum()), "drought_months": float(flags["drought"].sum())}
    return {"p": P, "stats": stats_}


def observed_stats(sti, sapei, scdhi, dates_ns, win) -> dict:
    dates = pd.DatetimeIndex(dates_ns)
    ser = {k: pd.Series(v, index=dates) for k, v in (("sti", sti), ("sapei", sapei), ("scdhi", scdhi))}
    _, _, _, f = build_event_masks(ser["sti"], ser["sapei"], ser["scdhi"], window_index=dates[win["pos"]],
                                   return_flags=True)
    comp = f["compound"].values
    return {"compound_months": float(comp.sum()), "mean_compound_run": run_lengths(comp),
            "compound_heat_share": float((comp & f["heat"].values).sum() / max(comp.sum(), 1)),
            "compound_drought_share": float((comp & f["drought"].values).sum() / max(comp.sum(), 1)),
            "heat_months": float(f["heat"].sum()), "drought_months": float(f["drought"].sum())}


def placebo_positions(n_months: int, dates: pd.DatetimeIndex, last_end="2000-12-01", first_year: int = 1951):
    """Window start positions: Jan of each year from `first_year` whose n_months-window ends by `last_end`."""
    starts = np.flatnonzero((dates.month == 1) & (dates.year >= first_year))
    end_pos = int(np.flatnonzero(dates == pd.Timestamp(last_end))[0])
    return [int(s) for s in starts if s + n_months - 1 <= end_pos]


def _chunk(kind, draws, z, raw_params, dates_ns, windows, n_perm, seed):
    out = []
    for k in draws:
        res = {}
        if kind == "N2":
            rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(k + 1)[k])
            raw = simulate_raw(raw_params, rng)
            sim = [zscore_array(raw[:, i]) for i in range(3)]
        for wname, w in windows.items():
            if kind == "N1":
                pos = w["placebo_starts"][k] + np.arange(len(w["pos"]))
                res[wname] = evaluate_draw(z[0], z[1], z[2], dates_ns, {**w, "pos": pos}, n_perm, seed + 7919 * k)
            else:
                res[wname] = evaluate_draw(sim[0], sim[1], sim[2], dates_ns, w, n_perm, seed + 7919 * k)
        out.append(res)
    return out


def conditional_mask(compound_counts, observed: float, tol: float = 0.25) -> tuple:
    """Draws whose compound count is within +-tol of the observed one (bounds rounded: 63 -> 47..79, 53 -> 40..66)."""
    lo, hi = int(round((1 - tol) * observed)), int(round((1 + tol) * observed))
    c = np.asarray(compound_counts, dtype=float)
    return (c >= lo) & (c <= hi), (lo, hi)


def rate_cell(v: np.ndarray, alpha: float = ALPHA) -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    k = int((v < alpha).sum())
    lo, hi = wilson_interval(k, len(v))
    return {"rate": k / len(v) if len(v) else np.nan, "wilson_lo": lo, "wilson_hi": hi, "n_draws": int(len(v))}


def _run(kind, draws, z, params, dates_ns, windows, n_perm, seed, n_jobs, chunk):
    chunks = [draws[i:i + chunk] for i in range(0, len(draws), chunk)]
    res = Parallel(n_jobs=n_jobs)(delayed(_chunk)(kind, c, z, params, dates_ns, windows, n_perm, seed)
                                  for c in chunks)
    return [r for c in res for r in c]


def run_nullsim(z: np.ndarray, dates_ns: np.ndarray, windows: dict, raw_for_fit: np.ndarray | None,
                n_draws_n2: int = 1000, n_perm: int = 499, n_jobs: int = 1, seed: int = SEED,
                chunk: int = 5, min_cond: int = 200, max_total: int = 2000) -> dict:
    """
    z: (3, T) real z-scored statewide [STI, SAPEI3, SCDHI3], unsmoothed, on `dates_ns`.
    windows: {name: {"pos": real positions of the outcome window, "index_ns", "Y", "names", "trends"}}.

    N1 is descriptive. N2 reports an unconditional rejection rate (the first `n_draws_n2` draws) and a rate
    conditional on the compound count being within +-25% of the observed one; if fewer than `min_cond` draws
    qualify, extra N2 draws (up to `max_total` in all) are added to the conditional pool only.
    Returns {"rejection", "diagnostics", "n_draws", "conditional", "var_order"}.
    """
    dates = pd.DatetimeIndex(dates_ns)
    params = fit_var_simulator(raw_for_fit, dates.month.values - 1)
    wN1 = {name: {**w, "placebo_starts": placebo_positions(len(w["pos"]), dates)} for name, w in windows.items()}
    n1_draws = {name: len(w["placebo_starts"]) for name, w in wN1.items()}
    obs = {name: observed_stats(z[0], z[1], z[2], dates_ns, w) for name, w in windows.items()}
    rows, diag, cond_info = [], [], {}

    pools = {}  # (kind, window) -> (base results, conditional pool of P arrays)
    for name in wN1:
        res = _run("N1", list(range(n1_draws[name])), z, None, dates_ns, {name: wN1[name]}, n_perm, seed,
                   n_jobs, chunk)
        pools[("N1", name)] = ([r[name] for r in res], None)
    base = _run("N2", list(range(n_draws_n2)), z, params, dates_ns, windows, n_perm, seed, n_jobs, chunk)
    for name, w in windows.items():
        lst = [r[name] for r in base]
        m, (lo, hi) = conditional_mask([r["stats"]["compound_months"] for r in lst], obs[name]["compound_months"])
        pool = [r["p"] for r, ok in zip(lst, m) if ok]
        total, extra = len(lst), 0
        while len(pool) < min_cond and total < max_total:
            k = min(max(200, min_cond), max_total - total)
            more = _run("N2", list(range(total, total + k)), z, params, dates_ns, {name: w}, n_perm, seed,
                        n_jobs, chunk)
            total += k
            extra += k
            mm, _ = conditional_mask([r[name]["stats"]["compound_months"] for r in more],
                                     obs[name]["compound_months"])
            pool += [r[name]["p"] for r, ok in zip(more, mm) if ok]
        cond_info[name] = {"compound_range": [lo, hi], "n_base_qualifying": int(m.sum()),
                           "n_extra_draws": extra, "n_conditional_pool": len(pool), "n_total_draws": total}
        pools[("N2", name)] = (lst, np.stack(pool) if pool else np.empty((0,) + lst[0]["p"].shape))

    for (kind, name), (lst, Pc) in pools.items():
        P = np.stack([r["p"] for r in lst])
        for j, nm in enumerate(windows[name]["names"]):
            for ti, term in enumerate(TERMS):
                for pi, pt in enumerate(P_TYPES):
                    row = {"generator": kind, "window": name, "series": nm, "term": term, "p_type": pt,
                           **rate_cell(P[:, j, ti, pi])}
                    c = rate_cell(Pc[:, j, ti, pi]) if Pc is not None else \
                        {"rate": np.nan, "wilson_lo": np.nan, "wilson_hi": np.nan, "n_draws": 0}
                    row.update({"cond_" + k: v for k, v in c.items()})
                    rows.append(row)
        for stat in lst[0]["stats"]:
            v = np.array([r["stats"][stat] for r in lst], dtype=float)
            v = v[np.isfinite(v)]
            q = np.percentile(v, [5, 50, 95]) if len(v) else [np.nan] * 3
            o = obs[name][stat]
            diag.append({"generator": kind, "window": name, "statistic": stat, "observed": o, "sim_p05": q[0],
                         "sim_median": q[1], "sim_p95": q[2], "outside_5_95": bool(o < q[0] or o > q[2]),
                         "n_draws": int(len(lst))})
    return {"rejection": pd.DataFrame(rows), "diagnostics": pd.DataFrame(diag),
            "n_draws": {"N1": n1_draws, "N2": n_draws_n2}, "conditional": cond_info, "var_order": params["p"]}
