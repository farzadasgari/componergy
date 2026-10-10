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


def test_temperature_dependence_detects_temperature_entering_the_drought_index():
    rng = np.random.default_rng(1)
    idx = _idx()
    sti = pd.Series(rng.normal(size=len(idx)), index=idx)
    spi = pd.Series(rng.normal(size=len(idx)), index=idx)
    sapei = 0.6 * spi - 0.6 * sti + 0.3 * rng.normal(size=len(idx))
    r = temperature_dependence(sti, sapei, spi)
    assert r["corr_STI_SAPEI"] < -0.5 and abs(r["corr_STI_SPI"]) < 0.1
    assert r["excess_r2_from_temperature"] > 0.25 and "twice" in r["verdict"]
    clean = temperature_dependence(sti, spi + 0.1 * rng.normal(size=len(idx)), spi)
    assert clean["excess_r2_from_temperature"] < 0.05 and "little evidence" in clean["verdict"]


def test_drought_agreement_counts_and_jaccard():
    rng = np.random.default_rng(2)
    idx = _idx()
    spi = pd.Series(rng.normal(size=len(idx)), index=idx)
    same = drought_definition_agreement(spi, spi)
    assert same["jaccard"] == pytest.approx(1.0) and same["SAPEI_only"] == 0
    other = drought_definition_agreement(pd.Series(rng.normal(size=len(idx)), index=idx), spi)
    assert other["jaccard"] < 0.3
