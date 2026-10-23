# src/componergy/review/episodes.py
"""Compound episodes and effective event counts: how many independent events are behind the event months."""

from __future__ import annotations

import numpy as np
import pandas as pd

from componergy.analysis.events import count_event_clusters
from componergy.review.inference import water_year_labels


def _runs(m: np.ndarray) -> list:
    """(first, last) positions of each run of True."""
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def compound_episodes(mask, merge_gap: int = 12) -> pd.DataFrame:
    """Runs of event months, merging runs separated by at most `merge_gap` non-event months."""
    s = pd.Series(mask).fillna(False).astype(bool)
    runs = _runs(s.values)
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = b
            merged[-1][2] += 1
        else:
            merged.append([a, b, 1])
    idx = s.index
    return pd.DataFrame({
        "start": [idx[a] for a, _, _ in merged], "end": [idx[b] for _, b, _ in merged],
        "n_months": [int(s.values[a:b + 1].sum()) for a, b, _ in merged],
        "n_runs": [r for _, _, r in merged], "span_months": [b - a + 1 for a, b, _ in merged]},
        columns=["start", "end", "n_months", "n_runs", "span_months"])


def episode_water_years(episodes: pd.DataFrame) -> list:
    """(first, last) water year (Oct-Sep, named by the year it ends in) spanned by each episode."""
    out = []
    for s, e in zip(episodes["start"], episodes["end"]):
        a, b = water_year_labels([s, e])
        out.append((int(a), int(b)))
    return out


def effective_event_counts(mask, merge_gap: int = 12) -> dict:
    """Runs, merged episodes, water years touched, lag-1 autocorrelation of the indicator and n_eff."""
    s = pd.Series(mask).fillna(False).astype(bool)
    x = s.values.astype(float)
    n = len(x)
    rho = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if x.std() > 0 and n > 2 else np.nan
    wy = water_year_labels(s.index)
    return {"n_months": int(n), "n_event_months": int(x.sum()), "n_runs": count_event_clusters(s),
            "n_episodes": int(len(compound_episodes(s, merge_gap))),
            "n_water_years": int(len(np.unique(wy[s.values]))), "rho_lag1": rho,
            "n_eff": float(n * (1 - rho) / (1 + rho)) if np.isfinite(rho) else np.nan}
