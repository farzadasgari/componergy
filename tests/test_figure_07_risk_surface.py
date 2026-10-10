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
