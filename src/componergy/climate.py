import xarray as xr
import numpy as np
from scipy.stats import norm, rankdata
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
from math import sin, cos, radians, pi

warnings.filterwarnings("ignore")


def hargreaves_pet(lat, doy, tmean, tmax, tmin):
    lat_rad = radians(lat)
    sol_dec = 0.409 * sin(2 * pi * doy / 365 - 1.39)
    sha = np.arccos(-np.tan(lat_rad) * np.tan(sol_dec))
    ra = 15.39 * (24 / pi) * sha * (0.0835 *
                                    sin(2 * pi * doy / 365 - 1.39) + 0.0007)
    trange = max(tmax - tmin, 0)
    pet_daily = 0.0023 * ra * (tmean + 17.8) * np.sqrt(trange)
    return pet_daily


def standardize(series):
    valid = ~np.isnan(series)
    if valid.sum() < 10:
        return np.full_like(series, np.nan)
    ranks = rankdata(series[valid], method='average')
    n = valid.sum()
    p = (ranks - 0.44) / (n + 0.12)
    p = np.clip(p, 1e-8, 1 - 1e-8)
    out = np.full_like(series, np.nan)
    out[valid] = norm.ppf(p)
    return out
