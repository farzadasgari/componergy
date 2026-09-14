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
