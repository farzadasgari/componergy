import numpy as np
import pandas as pd

from componergy.review.episodes import compound_episodes, effective_event_counts, episode_water_years


def _mask(runs, n=120):
    idx = pd.date_range("2001-01-01", periods=n, freq="MS")
    m = np.zeros(n, bool)
    for a, b in runs:
        m[a:b + 1] = True
    return pd.Series(m, index=idx)


def test_runs_separated_by_5_merge_and_by_15_do_not():
    # run A 10-12, gap of 5, run B 18-19, gap of 15, run C 35-36
    ep = compound_episodes(_mask([(10, 12), (18, 19), (35, 36)]))
    assert len(ep) == 2 and ep["n_runs"].tolist() == [2, 1] and ep["n_months"].tolist() == [5, 2]
    assert ep.loc[0, "start"] == pd.Timestamp("2001-11-01") and ep.loc[0, "end"] == pd.Timestamp("2002-08-01")
    assert episode_water_years(ep)[0] == (2002, 2002)


def test_effective_counts():
    c = effective_event_counts(_mask([(10, 12), (18, 19), (35, 36)]))
    assert c["n_runs"] == 3 and c["n_episodes"] == 2 and c["n_event_months"] == 7
    assert 0 < c["rho_lag1"] < 1 and c["n_eff"] < c["n_months"]
