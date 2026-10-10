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


def test_too_short_an_overlap_is_refused_rather_than_summarised():
    a = _seasonal()
    r = compare_demand_sources(a.iloc[:12], a.iloc[:12])
    assert r["n_months"] == 12 and "no meaningful comparison" in r["verdict"]


def test_statewide_monthly_mwh_sums_every_county_and_sector():
    dates = pd.date_range("2010-01-01", periods=3, freq="MS")
    rows = [{"Date": d, "CountyName": c, "Sector": s, "Consumption (MWh)": float(v)}
            for d in dates for c, v in (("A", 10), ("B", 20)) for s in ("Residential", "Commercial")]
    out = statewide_monthly_mwh(pd.DataFrame(rows))
    assert list(out.values) == [60.0, 60.0, 60.0]
    assert out.name == "total_sales" and out.index.name is None
