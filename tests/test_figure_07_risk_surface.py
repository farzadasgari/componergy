import numpy as np
import pandas as pd
import xarray as xr

from componergy.figures.risk_surface import (
    compute_gridded_vulnerability,
    compute_risk_surface,
    compute_return_period,
    region_abbrev,
    compute_regional_evi
)


def test_region_abbrev_uses_first_letters_of_up_to_three_words():
    assert region_abbrev("San Joaquin Valley") == "SJV"
    assert region_abbrev("North Coast") == "NC"
    assert region_abbrev("Sacramento-Delta") == "SD"


def test_compute_gridded_vulnerability_isolates_known_localized_signal():
    n = 100
    times = pd.date_range("2008-01-01", periods=n, freq="MS")
    lats = np.linspace(33, 41, 4)
    lons = np.linspace(-123, -115, 4)

    rng = np.random.default_rng(0)
    sti_data = rng.normal(0, 0.2, (n, 4, 4))
    sapei_data = rng.normal(0, 0.2, (n, 4, 4))
    scdhi_data = rng.normal(0, 0.2, (n, 4, 4))

    heat_months = np.zeros(n, dtype=bool)
    heat_months[:15] = True
    sti_data[heat_months, 0, 0] += 3.0

    drought_months = np.zeros(n, dtype=bool)
    drought_months[15:25] = True
    compound_months = np.zeros(n, dtype=bool)
    compound_months[25:35] = True

    indices_ds = xr.Dataset(
        {"sti": (("time", "lat", "lon"), sti_data),
         "sapei_3m": (("time", "lat", "lon"), sapei_data),
         "scdhi_3m": (("time", "lat", "lon"), scdhi_data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )

    heat_mask = pd.Series(heat_months, index=times)
    drought_mask = pd.Series(drought_months, index=times)
    compound_mask = pd.Series(compound_months, index=times)
    normal_mask = ~(heat_mask | drought_mask | compound_mask)

    data = {
        "indices_ds": indices_ds,
        "event_masks": {"Heatwave": heat_mask, "Drought": drought_mask, "Compound": compound_mask},
        "normal": normal_mask,
    }

    gridded, is_significant = compute_gridded_vulnerability(data)
    assert gridded.isel(lat=0, lon=0) > 0.5
    assert abs(float(gridded.isel(lat=1, lon=1))) < 0.3


def test_compute_risk_surface_product_is_correct():
    idx = pd.date_range("2010-01-01", periods=10, freq="MS")
    data = {
        "scdhi": pd.Series([1, -1, 2, -2, 0, 0.5, -0.5, 1.5, -1.5, 3], index=idx),
        "demand_state_z": pd.Series([1, 1, -1, -1, 2, -2, 0.5, 0.5, -0.5, 1], index=idx),
    }
    risk_df = compute_risk_surface(data)

    assert risk_df["System_Risk"].iloc[0] == (-1) * 1
    assert risk_df["System_Risk"].iloc[2] == (-2) * (-1)


def test_compute_return_period_weibull_plotting_position():
    risk = pd.Series(np.arange(1, 13), index=pd.date_range("2020-01-01", periods=12, freq="MS"))
    rp = compute_return_period(risk)

    assert rp["System_Risk"].iloc[0] == 12
    assert abs(rp["Return_Period_Years"].iloc[0] - (13 / 1) / 12) < 1e-9
    assert rp["Return_Period_Years"].is_monotonic_decreasing


def test_compute_regional_evi_averages_delta_z_across_events():
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2008-01-01", periods=n, freq="MS")

    heat_mask = pd.Series([False] * n, index=idx)
    heat_mask.iloc[:20] = True
    drought_mask = pd.Series([False] * n, index=idx)
    drought_mask.iloc[20:40] = True
    compound_mask = pd.Series([False] * n, index=idx)
    compound_mask.iloc[40:55] = True
    normal_mask = ~(heat_mask | drought_mask | compound_mask)

    region_a = pd.Series(rng.normal(0, 1, n), index=idx)
    region_a.iloc[:20] += 2.0
    region_a.iloc[20:40] -= 2.0

    electricity_z = pd.DataFrame({"RegionA": region_a})
    data = {
        "electricity_raw": electricity_z,
        "electricity_z": electricity_z,
        "event_masks": {"Heatwave": heat_mask, "Drought": drought_mask, "Compound": compound_mask},
        "normal": normal_mask,
    }

    evi, comp = compute_regional_evi(data)
    assert "q_value" in comp.columns
    assert abs(evi.loc["RegionA"]) < 1.0
