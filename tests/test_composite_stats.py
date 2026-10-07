import numpy as np
import pandas as pd
import xarray as xr

from componergy.figures.composite_stats import (
    series_to_data_array, welch_ttest, compute_composite_table, best_lag_slope,
    benjamini_hochberg_fdr, compute_incremental_r2, remove_linear_trend, apply_fdr_to_grid,
)
from componergy.analysis.events import classify_heatwave


def test_series_to_data_array_bridge_matches_manual_percentile_classification():
    rng = np.random.default_rng(0)
    times = pd.date_range("1951-01-01", periods=900, freq="MS")
    sti_series = pd.Series(rng.normal(0, 1, 900), index=times)

    sti_da = series_to_data_array(sti_series)
    heat_series = classify_heatwave(sti_da).to_series()

    manual_threshold = np.percentile(sti_series.values, 90)
    manual_heat = sti_series > manual_threshold
    assert (heat_series.values == manual_heat.values).all()


def test_welch_ttest_matches_scipy_directly():
    from scipy import stats as sstats
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(5, 1, 100))
    b = pd.Series(rng.normal(0, 1, 100))

    t, p = welch_ttest(a, b)
    t_ref, p_ref = sstats.ttest_ind(a, b, equal_var=False)
    assert abs(t - t_ref) < 1e-9
    assert abs(p - p_ref) < 1e-9


def test_welch_ttest_returns_nan_for_small_samples():
    t, p = welch_ttest(pd.Series([1, 2, 3]), pd.Series([1, 2, 3, 4, 5]))
    assert np.isnan(t) and np.isnan(p)


def test_composite_table_matches_manual_mean_difference():
    rng = np.random.default_rng(1)
    n = 200
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    signal_df = pd.DataFrame({"SourceA": rng.normal(0, 1, n)}, index=idx)
    event_mask = pd.Series([True] * 30 + [False] * (n - 30), index=idx)
    normal_mask = pd.Series([False] * 30 + [True] * (n - 30), index=idx)

    table = compute_composite_table(signal_df, {"TestEvent": event_mask}, normal_mask)
    row = table.iloc[0]

    expected_diff = signal_df.loc[event_mask, "SourceA"].mean() - signal_df.loc[normal_mask, "SourceA"].mean()
    assert abs(row["Difference"] - expected_diff) < 1e-9
    assert row["N_event"] == 30
    assert row["N_normal"] == n - 30


def test_best_lag_slope_recovers_known_lag_and_slope():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(0, 1, 300))
    true_lag, true_slope = 3, 2.5
    y = pd.Series(np.full(300, np.nan))
    y.iloc[true_lag:] = true_slope * x.iloc[:-true_lag].values + rng.normal(0, 0.05, 300 - true_lag)

    best = best_lag_slope(x, y, lags=range(-6, 7), min_n=60)

    assert best["lag"] == true_lag
    assert abs(best["slope"] - true_slope) < 0.1
    assert best["r2"] > 0.95


def test_best_lag_slope_p_value_matches_scipy_linregress():
    from scipy import stats as sstats
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(0, 1, 200))
    y = pd.Series(0.05 * x.values + rng.normal(0, 1, 200))

    best = best_lag_slope(x, y, lags=[0], min_n=60)
    ref = sstats.linregress(x.values, y.values)

    assert abs(best["p_value"] - ref.pvalue) < 1e-6


def test_benjamini_hochberg_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p_values = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.9])

    our_q = benjamini_hochberg_fdr(p_values)
    _, ref_q, _, _ = multipletests(p_values, method="fdr_bh")

    assert np.allclose(our_q, ref_q, atol=1e-10)


def test_benjamini_hochberg_passes_nan_through_without_corrupting_others():
    p_with_nan = np.array([0.01, np.nan, 0.03, 0.5])
    q = benjamini_hochberg_fdr(p_with_nan)

    assert np.isnan(q[1])
    assert np.isfinite(q[0]) and np.isfinite(q[2]) and np.isfinite(q[3])


def test_compute_incremental_r2_matches_statsmodels_nested_f_test():
    import statsmodels.api as sm

    rng = np.random.default_rng(5)
    n = 300
    x_base = rng.normal(0, 1, n)
    x_additional = 0.6 * x_base + rng.normal(0, 1, n)
    y = 1.5 * x_base + 0.8 * x_additional + rng.normal(0, 1, n)

    result = compute_incremental_r2(y, x_base, x_additional)

    X_reduced = sm.add_constant(x_base)
    X_full = sm.add_constant(np.column_stack([x_base, x_additional]))
    model_reduced = sm.OLS(y, X_reduced).fit()
    model_full = sm.OLS(y, X_full).fit()
    f_val, p_val, _ = model_full.compare_f_test(model_reduced)

    assert abs(result["f_stat"] - f_val) < 1e-6
    assert abs(result["p_value"] - p_val) / max(p_val, 1e-300) < 1e-3
    assert abs(result["r2_full"] - model_full.rsquared) < 1e-9
    assert abs(result["r2_reduced"] - model_reduced.rsquared) < 1e-9


def test_compute_incremental_r2_zero_when_additional_predictor_adds_nothing():
    rng = np.random.default_rng(9)
    n = 200
    x_base = rng.normal(0, 1, n)
    x_additional = rng.normal(0, 1, n)  # independent noise, no real relationship to y
    y = 2.0 * x_base + rng.normal(0, 0.5, n)

    result = compute_incremental_r2(y, x_base, x_additional)

    assert result["delta_r2"] < 0.02
    assert result["p_value"] > 0.05


def test_remove_linear_trend_recovers_known_slope_and_leaves_no_residual_trend():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2001-01-01", periods=300, freq="MS")
    years_frac = idx.year.to_numpy(dtype=float) + (idx.month.to_numpy(dtype=float) - 0.5) / 12
    true_slope, true_intercept = 0.8, -1500
    series = pd.Series(true_intercept + true_slope * years_frac + rng.normal(0, 1, 300), index=idx)

    residuals = remove_linear_trend(series)

    assert abs(residuals.mean()) < 1e-6
    remaining_slope = np.polyfit(years_frac, residuals.values, 1)[0]
    assert abs(remaining_slope) < 0.01
    assert abs(residuals.std() - 1.0) < 0.15


def test_apply_fdr_to_grid_distinguishes_signal_from_noise_and_preserves_nan():
    rng = np.random.default_rng(0)
    pvals = rng.uniform(0.4, 1.0, (10, 10))
    pvals[:5, :] = rng.uniform(0, 0.001, (5, 10))
    pvals[8:, 8:] = np.nan
    da = xr.DataArray(pvals, dims=("lat", "lon"), coords={"lat": np.arange(10), "lon": np.arange(10)})

    q = apply_fdr_to_grid(da)

    assert (q.isel(lat=slice(0, 5)).values < 0.05).mean() > 0.8
    assert (q.isel(lat=slice(5, 8)).values < 0.05).mean() < 0.3
    assert np.isnan(q.isel(lat=9, lon=9).values)


def test_vectorized_welch_ttest_grid_matches_scipy_at_every_cell():
    from componergy.figures.composite_stats import vectorized_welch_ttest_grid
    from scipy import stats as sstats

    n = 200
    lats = np.linspace(33, 41, 4)
    lons = np.linspace(-123, -115, 4)
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, (n, 4, 4))

    event_months = np.zeros(n, dtype=bool)
    event_months[:30] = True
    data[event_months, 0, 0] += 3.0
    normal_months = ~event_months

    da = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": np.arange(n), "lat": lats, "lon": lons})
    mask_event = xr.DataArray(event_months, dims="time", coords={"time": np.arange(n)})
    mask_normal = xr.DataArray(normal_months, dims="time", coords={"time": np.arange(n)})

    pvals = vectorized_welch_ttest_grid(da, mask_event, mask_normal)

    for i in range(4):
        for j in range(4):
            ev = data[event_months, i, j]
            nm = data[normal_months, i, j]
            _, ref_p = sstats.ttest_ind(ev, nm, equal_var=False)
            assert abs(float(pvals.isel(lat=i, lon=j)) - ref_p) < 1e-8
