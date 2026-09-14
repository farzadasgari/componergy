import numpy as np
import pandas as pd
import xarray as xr

from componergy.figures.trend_stats import (
    compute_area_weights,
    compute_statewide_mean,
    compute_footprint_fraction,
    annualize_monthly,
    compute_trend_per_decade,
    compute_linear_trend_with_ci,
)

def test_area_weights_match_manual_cosine_calc():
    lats = xr.DataArray([0.0, 30.0, 60.0, 90.0], dims="lat")
    w = compute_area_weights(lats)
    expected = np.cos(np.deg2rad(lats.values))
    expected = expected / expected.mean()
    assert np.allclose(w.values, expected)