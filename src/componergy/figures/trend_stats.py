from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


def compute_area_weights(lat: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(lat))
    return weights / weights.mean()
