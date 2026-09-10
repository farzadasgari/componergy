import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm

from componergy.indices.sapei import (
    add_sapei,
    water_balance,
    antecedent_water_balance,
    SAPEI_TIMESCALES_MONTHS,
)
from componergy.indices.loglogistic import fit_loglogistic_lmoments, loglogistic_cdf
