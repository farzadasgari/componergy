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
