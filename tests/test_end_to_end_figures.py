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
