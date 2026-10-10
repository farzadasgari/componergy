import numpy as np
import pandas as pd

from componergy.analysis.demand_crosscheck import compare_demand_sources
from componergy.preprocessing.consumption_cec import statewide_monthly_mwh


def _seasonal(n=204, seed=0, noise=0.02):
    idx = pd.date_range("2008-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    season = 1.0 + 0.25 * np.sin(2 * np.pi * (idx.month.values - 4) / 12)
    return pd.Series(20000 * season * (1 + rng.normal(0, noise, n)), index=idx)


def test_identical_series_agree_perfectly():
    a = _seasonal()
    r = compare_demand_sources(a, a * 1.07)
    assert r["corr_deseasonalized_anomalies"] > 0.999
    assert abs(r["level_ratio_mean"] - 1 / 1.07) < 1e-9
    assert "close agreement" in r["verdict"]


def test_shared_seasonality_alone_does_not_count_as_agreement():
    a, b = _seasonal(seed=1, noise=0.03), _seasonal(seed=2, noise=0.03)
    r = compare_demand_sources(a, b)
    assert r["corr_raw_series"] > 0.95
    assert r["corr_deseasonalized_anomalies"] < 0.4
    assert "weak agreement" in r["verdict"]
