from componergy.climate import (
    hargreaves_pet,
    standardize,
)
from componergy.paths import (
    NOAA_MONTHLY_FILE,
    WARM_SEASON_FILE,
    INDICES_DIR,
)
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from scipy.stats import norm, rankdata
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


INDICES_DIR.mkdir(parents=True, exist_ok=True)


def compute_scdhi_monthly(prcp, tmean, tmax, tmin, lat, month):
    doy = 30.5 * (month - 1) + 15
    pet = np.array([
        hargreaves_pet(lat, doy, tmean[i], tmax[i], tmin[i]) * 30.5
        if not np.isnan(tmean[i]) else np.nan
        for i in range(len(tmean))
    ])

    valid = ~(np.isnan(prcp) | np.isnan(pet) | np.isnan(tmean))
    if valid.sum() < 10:
        return np.full_like(prcp, np.nan)

    g1 = (rankdata(prcp[valid] - pet[valid], 'average') -
          0.44) / (valid.sum() + 0.12)
    g2 = (rankdata(tmean[valid], 'average') - 0.44) / (valid.sum() + 0.12)
    g1 = np.clip(g1, 1e-8, 1-1e-8)
    g2 = np.clip(g2, 1e-8, 1-1e-8)

    x = g1 / g2
    scdhi = standardize(x)

    result = np.full_like(prcp, np.nan)
    result[valid] = scdhi
    return result


def process_cell(i_lat, i_lon):
    cell = monthly_ds.sel(lat=lats[i_lat], lon=lons[i_lon], method='nearest')[
        ['prcp', 'tmean', 'tmax', 'tmin']]
    result = np.full((75, 12, 5), np.nan)

    for month in range(1, 13):
        data = cell.where(cell.time.dt.month == month, drop=True)
        if len(data.time) == 0:
            continue

        prcp = np.array([float(x) if not np.isnan(
            x) else np.nan for x in data.prcp.values])
        tmean = np.array([float(x) if not np.isnan(
            x) else np.nan for x in data.tmean.values])
        tmax = np.array([float(x) if not np.isnan(
            x) else np.nan for x in data.tmax.values])
        tmin = np.array([float(x) if not np.isnan(
            x) else np.nan for x in data.tmin.values])

        years_idx = data.time.dt.year.values - 1951

        if len(prcp) < 10:
            continue

        scdhi = compute_scdhi_monthly(
            prcp, tmean, tmax, tmin, lats[i_lat], month)

        valid = ~np.isnan(scdhi)
        valid_years = years_idx[valid]
        valid_scdhi = scdhi[valid]

        if len(valid_years) == 0:
            continue

        result[valid_years, month-1, 0] = prcp[valid]
        result[valid_years, month-1, 1] = tmean[valid]
        result[valid_years, month-1, 2] = standardize(prcp - np.array([hargreaves_pet(
            lats[i_lat], 30.5*(month-1)+15, tmean[i], tmax[i], tmin[i]) for i in range(len(tmean))]))[valid]
        result[valid_years, month-1, 3] = standardize(tmean)[valid]
        result[valid_years, month-1, 4] = valid_scdhi

    return i_lat, i_lon, result


monthly_ds = xr.open_dataset(NOAA_MONTHLY_FILE).load()
warm_da = xr.open_dataarray(WARM_SEASON_FILE)

mask = ~np.isnan(warm_da).any(dim='window')
lat_idx, lon_idx = np.where(mask)
lats = monthly_ds.lat.values
lons = monthly_ds.lon.values


results = Parallel(n_jobs=-1, backend='loky')(
    delayed(process_cell)(i, j) for i, j in tqdm(zip(lat_idx, lon_idx), total=len(lat_idx), desc="SCDHI")
)

ny, nm = 75, 12
final = {v: np.full((ny, nm, len(lats), len(lons)), np.nan)
         for v in ['prcp', 'tmean', 'sapei', 'sti', 'scdhi']}

for i_lat, i_lon, arr in results:
    for v in range(5):
        name = ['prcp', 'tmean', 'sapei', 'sti', 'scdhi'][v]
        final[name][:, :, i_lat, i_lon] = arr[:, :, v]

ds = xr.Dataset(
    {k: (['year', 'month', 'lat', 'lon'], v) for k, v in final.items()},
    coords={'year': range(1951, 2026), 'month': range(
        1, 13), 'lat': lats, 'lon': lons}
)

OUTPUT_FILE = INDICES_DIR / "scdhi-monthly-ca-1951-2025.nc"

ds.to_netcdf(
    OUTPUT_FILE,
    encoding={v: {"zlib": True, "complevel": 5} for v in ds.data_vars},
)
