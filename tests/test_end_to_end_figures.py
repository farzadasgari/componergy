"""Run each composite figure's real main() on synthetic inputs and check what it writes."""

import json
from contextlib import ExitStack
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
import pytest

from tests import synthetic_inputs as si


@pytest.fixture(autouse=True)
def _fast_rendering():
    old = matplotlib.rcParams["savefig.dpi"]
    matplotlib.rcParams["savefig.dpi"] = 40
    yield
    matplotlib.rcParams["savefig.dpi"] = old


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    ds = si.make_indices()
    ds.to_netcdf(d / "indices.nc")
    gen = si.make_generation()
    gen.to_csv(d / "generation.csv")
    regions, counties = si.make_geography()
    return {"dir": d, "indices": d / "indices.nc", "generation_csv": d / "generation.csv", "generation": gen,
            "consumption": si.make_consumption_long(), "regions": regions, "counties": counties}


def _paths(stack, inputs, out):
    stack.enter_context(patch("componergy.paths.INDICES_FILE", inputs["indices"]))
    stack.enter_context(patch("componergy.paths.GENERATION_MONTHLY_FILE", inputs["generation_csv"]))
    stack.enter_context(patch("componergy.paths.FIGURES_DIR", out))
    stack.enter_context(patch("componergy.paths.COUNTY_CONSUMPTION_XLSX", inputs["dir"] / "x.xlsx"))


def _geo(stack, inputs):
    stack.enter_context(patch("componergy.download.climate_regions.get_california_climate_regions",
                              return_value=inputs["regions"]))
    stack.enter_context(patch("componergy.download.boundaries.get_california_counties",
                              return_value=inputs["counties"]))
    stack.enter_context(patch("componergy.preprocessing.consumption_cec.load_raw_consumption",
                              return_value=inputs["consumption"]))


COMPOSITE_TABLES = ("amplification", "superadditivity", "placebo", "specification_robustness")


def _check_inference_outputs(out, prefix):
    for name in COMPOSITE_TABLES:
        assert (out / f"{prefix}_{name}.csv").exists(), f"{prefix}_{name}.csv missing"
    rob = pd.read_csv(out / f"{prefix}_specification_robustness.csv")
    specs = set(rob["Specification"])
    assert any("naive composite" in s for s in specs)
    assert {"month effects only", "month effects + linear trend",
            "month + year fixed effects"} <= specs
    amp = pd.read_csv(out / f"{prefix}_amplification.csv")
    assert {"Amplification_contrast", "CI_lo", "CI_hi", "p_value", "p_exact", "p_hac", "q_value"} <= set(amp.columns)


def _check_log(out, stem):
    log = json.load(open(out / f"{stem}.json"))
    ev = log["Event definition (shared across Figures 2 and 4-7)"]
    assert ev["baseline"] == "full_record"
    assert {"episodes", "overlap", "counts_raw", "thresholds"} <= set(ev)
    assert any("Amplification contrast" in k for k in log)
    return log


def test_figure_02_end_to_end(tmp_path, inputs):
    import componergy.figures.generation as f2
    with ExitStack() as st:
        _paths(st, inputs, tmp_path)
        st.enter_context(patch.object(f2, "INDICES_FILE", inputs["indices"]))
        st.enter_context(patch.object(f2, "GENERATION_MONTHLY_FILE", inputs["generation_csv"]))
        st.enter_context(patch.object(f2, "FIGURES_DIR", tmp_path))
        f2.main()
    out = tmp_path / "figure_02"
    comp = pd.read_csv(out / "figure_02_composite.csv")
    assert {"total_share", "deseasonalized_share", "detrended_share", "adjusted_share"} <= set(comp["Mode"])
    lag = pd.read_csv(out / "figure_02_lag_scan.csv")
    assert {"adjusted_share"} <= set(lag["Mode"]) and {"p_value", "p_ols_naive", "q_value", "n_perm"} <= set(
        lag.columns)
    assert (lag["p_value"] >= 1 / 1000).all()
    _check_inference_outputs(out, "figure_02")
    _check_log(out, "figure_02_generation_response")


def test_figure_04_end_to_end(tmp_path, inputs):
    import componergy.figures.zonal_consumption as f4
    with ExitStack() as st:
        _paths(st, inputs, tmp_path)
        _geo(st, inputs)
        f4.main()
    out = tmp_path / "figure_04"
    assert (out / "figure_04_zonal_consumption.jpg").stat().st_size > 5_000
    comp = pd.read_csv(out / "figure_04_event_composites.csv")
    assert {"Region", "Event", "Delta_pct", "Delta_z", "q_value", "p_exact", "Delta_native"} <= set(comp.columns)
    assert set(comp["Event"]) == {"Heatwave", "Drought", "Compound"}
    _check_inference_outputs(out, "figure_04")
    keys = pd.read_csv(out / "figure_04_region_abbreviations.csv")
    assert keys["Abbreviation"].is_unique and set(keys["Region"]) == set(si.REGION_NAMES)
    _check_log(out, "figure_04_zonal_consumption")


def test_figure_05_end_to_end(tmp_path, inputs):
    import componergy.figures.sector as f5
    with ExitStack() as st:
        _paths(st, inputs, tmp_path)
        _geo(st, inputs)
        f5.main()
    out = tmp_path / "figure_05"
    comp = pd.read_csv(out / "figure_05_sector_composites.csv")
    assert {"Sector", "Delta_GWh", "Delta_pct", "Delta_z", "q_value"} <= set(comp.columns)
    assert np.allclose(comp["Delta_GWh"], comp["Delta_native"])
    _check_inference_outputs(out, "figure_05")
    _check_log(out, "figure_05_sector_fingerprint")


def test_figure_06_end_to_end_uses_cec_by_default_and_runs_the_alternative_source(tmp_path, inputs):
    import componergy.figures.imbalance as f6
    gen = inputs["generation"]
    sources = {"cec": si.make_demand_mwh(gen, seed=10, start="2008-01-01", end="2024-12-01"),
               "eia861m": si.make_demand_mwh(gen, seed=11)}
    with ExitStack() as st:
        _paths(st, inputs, tmp_path)
        st.enter_context(patch.object(f6, "load_demand_sources", lambda: sources))
        f6.main()
    out = tmp_path / "figure_06"
    log = _check_log(out, "figure_06_imbalance")
    assert log["Overview"]["demand_source"] == "cec"
    assert log["Overview"]["period_start"] == "2008-01-01" and log["Overview"]["period_end"] == "2024-12-01"
    assert (out / "figure_06_event_metrics_eia861m_demand.csv").exists()
    assert set(log["System boundary validation (per demand source)"]) == {"cec", "eia861m"}
    assert "verdict" in log["Demand source cross-check (CEC vs EIA-861M)"]
    m = pd.read_csv(out / "figure_06_event_metrics.csv")
    assert {"RAF", "RAF_unadjusted", "RAF_p_shift_upper", "Delta_GWh", "q_value"} <= set(m.columns)
    _check_inference_outputs(out, "figure_06")
