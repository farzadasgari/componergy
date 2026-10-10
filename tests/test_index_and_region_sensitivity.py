import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import box

from componergy.analysis.index_diagnostics import (
    standardized_precip_index, temperature_dependence, drought_definition_agreement,
)
from componergy.analysis.regions import county_region_shares, county_split_summary, allocation_sensitivity


def _idx(n=600):
    return pd.date_range("1951-01-01", periods=n, freq="MS")


def test_spi_is_approximately_standard_normal_within_every_calendar_month():
    rng = np.random.default_rng(0)
    idx = _idx()
    prcp = pd.Series(rng.gamma(2.0, 30.0, len(idx)) * (1 + 0.8 * np.cos(2 * np.pi * idx.month / 12)), index=idx)
    spi = standardized_precip_index(prcp, scale=3)
    assert spi.iloc[:2].isna().all()
    by_month = spi.groupby(spi.index.month)
    assert np.allclose(by_month.mean().values, 0.0, atol=0.05)
    assert np.allclose(by_month.std().values, 1.0, atol=0.12)
