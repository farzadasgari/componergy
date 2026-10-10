import numpy as np
import pandas as pd
import xarray as xr

from componergy.figures.risk_surface import (
    compute_gridded_vulnerability, compute_risk_surface, compute_return_period, region_abbrev,
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
