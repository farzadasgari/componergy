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
