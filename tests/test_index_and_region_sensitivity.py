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


def _geo():
    regions = gpd.GeoDataFrame({"Region": ["West", "East"]},
                               geometry=[box(-122, 36, -120, 38), box(-120, 36, -118, 38)],
                               crs="EPSG:4326")
    counties = gpd.GeoDataFrame({"CountyName": ["INSIDE", "STRADDLE"]},
                                geometry=[box(-121.8, 36.5, -121.2, 37.0), box(-120.6, 37.0, -119.6, 37.5)],
                                crs="EPSG:4326")
    return counties, regions


def test_county_region_shares_sum_to_one_and_identify_a_straddling_county():
    counties, regions = _geo()
    shares = county_region_shares(counties, regions)
    assert np.allclose(shares.groupby("CountyName")["share"].sum().values, 1.0, atol=1e-6)
    summary = county_split_summary(shares).set_index("CountyName")
    assert summary.loc["INSIDE", "N_regions_touched"] == 1 and not summary.loc["INSIDE", "Split"]
    assert summary.loc["STRADDLE", "N_regions_touched"] == 2
    assert 0.55 < summary.loc["STRADDLE", "Majority_share"] < 0.65 and summary.loc["STRADDLE", "Split"]


def test_allocation_sensitivity_measures_how_much_consumption_changes_region():
    counties, regions = _geo()
    shares = county_region_shares(counties, regions)
    dates = pd.date_range("2008-01-01", periods=60, freq="MS")
    rng = np.random.default_rng(3)
    rows = [{"Date": d, "CountyName": c, "Sector": "Residential", "Consumption (MWh)": v}
            for d in dates for c, v in (("INSIDE", 100 + rng.normal()), ("STRADDLE", 100 + rng.normal()))]
    r = allocation_sensitivity(pd.DataFrame(rows), shares)
    assert 0.15 < r["share_of_consumption_changing_region"] < 0.25
    assert set(r["per_region"]["Region"]) == {"West", "East"}
