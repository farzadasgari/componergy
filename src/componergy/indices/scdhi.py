from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm import tqdm

from componergy.indices.copulas import FAMILIES, compare_families
from componergy.indices.gringorten import gringorten_standardize

MIN_SAMPLES_FOR_COPULA = 30
POOLING_SAMPLE_CELLS = 200