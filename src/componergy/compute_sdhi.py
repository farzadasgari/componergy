from componergy.climate import (
    hargreaves_pet,
    standardize,
)
from componergy.paths import (
    NOAA_MONTHLY_FILE,
    INDICES_DIR,
    WARM_SEASON_FILE,
)
import sys
import warnings
import numpy as np
import xarray as xr
from scipy.stats import norm, rankdata
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

INDICES_DIR.mkdir(parents=True, exist_ok=True)

monthly_ds = xr.open_dataset(NOAA_MONTHLY_FILE)
warm_da = xr.open_dataarray(WARM_SEASON_FILE)

mask = ~np.isnan(warm_da).any(dim='window')
lat_idxs, lon_idxs = np.where(mask)
n_cells = len(lat_idxs)

lats = monthly_ds.lat.values
lons = monthly_ds.lon.values

years = np.arange(1951, 2026)
n_years = len(years)

months = np.arange(1, 13)
n_months = len(months)

vars_to_save = [
    'prcp', 'tmean', 'tmax', 'tmin', 'pet',
    'spi', 'sti', 'spei',
    'x1', 'sdhi1', 'x2', 'sdhi2'
]
data_dict = {v: np.full((n_years, n_months, len(
    lats), len(lons)), np.nan) for v in vars_to_save}

for c_idx in tqdm(range(n_cells), desc="Cells"):
    i_lat = lat_idxs[c_idx]
    i_lon = lon_idxs[c_idx]
    lat = lats[i_lat]
    lon = lons[i_lon]

    cell = monthly_ds[['prcp', 'tmean', 'tmax', 'tmin']].sel(
        lat=lat, lon=lon, method='nearest')

    for m in months:
        m_p = []
        m_tmean = []
        m_tmax = []
        m_tmin = []
        m_pet = []

        for year in years:
            ydata = cell.sel(time=cell['time.year'] == year)
            month_data = ydata.sel(time=ydata['time.month'] == m)
            if month_data.time.size == 0:
                m_p.append(np.nan)
                m_tmean.append(np.nan)
                m_tmax.append(np.nan)
                m_tmin.append(np.nan)
                m_pet.append(np.nan)
                continue

            p_val = month_data['prcp'].values.item()
            tmean_val = month_data['tmean'].values.item()
            tmax_val = month_data['tmax'].values.item()
            tmin_val = month_data['tmin'].values.item()

            doy = 31 * (m - 1) + 15
            pet_month = hargreaves_pet(
                lat,
                doy,
                tmean_val,
                tmax_val,
                tmin_val,
            ) * month_data["time"].dt.daysinmonth.values.item()

            m_p.append(p_val)
            m_tmean.append(tmean_val)
            m_tmax.append(tmax_val)
            m_tmin.append(tmin_val)
            m_pet.append(pet_month)

        m_p = np.array(m_p)
        m_tmean = np.array(m_tmean)
        m_tmax = np.array(m_tmax)
        m_tmin = np.array(m_tmin)
        m_pet = np.array(m_pet)

        valid = ~(np.isnan(m_p) | np.isnan(m_tmean))

        if valid.sum() < 10:
            continue

        spi = standardize(m_p)
        sti = standardize(m_tmean)
        spei = standardize(m_p - m_pet)

        g1_p = (rankdata(m_p[valid], 'average') - 0.44) / (valid.sum() + 0.12)
        g2_t = (rankdata(m_tmean[valid], 'average') -
                0.44) / (valid.sum() + 0.12)
        g1_pe = (rankdata((m_p - m_pet)[valid],
                 'average') - 0.44) / (valid.sum() + 0.12)

        g1_p = np.clip(g1_p, 1e-6, 1 - 1e-6)
        g2_t = np.clip(g2_t, 1e-6, 1 - 1e-6)
        g1_pe = np.clip(g1_pe, 1e-6, 1 - 1e-6)

        x1 = g1_p / g2_t
        x2 = g1_pe / g2_t

        sdhi1 = norm.ppf((rankdata(x1, 'average') - 0.44) / (len(x1) + 0.12))
        sdhi2 = norm.ppf((rankdata(x2, 'average') - 0.44) / (len(x2) + 0.12))

        full_valid_idxs = np.where(valid)[0]
        data_dict['prcp'][full_valid_idxs, m-1, i_lat, i_lon] = m_p[valid]
        data_dict['tmean'][full_valid_idxs, m-1, i_lat, i_lon] = m_tmean[valid]
        data_dict['tmax'][full_valid_idxs, m-1, i_lat, i_lon] = m_tmax[valid]
        data_dict['tmin'][full_valid_idxs, m-1, i_lat, i_lon] = m_tmin[valid]
        data_dict['pet'][full_valid_idxs, m-1, i_lat, i_lon] = m_pet[valid]
        data_dict['spi'][full_valid_idxs, m-1, i_lat, i_lon] = spi
        data_dict['sti'][full_valid_idxs, m-1, i_lat, i_lon] = sti
        data_dict['spei'][full_valid_idxs, m-1, i_lat, i_lon] = spei
        data_dict['x1'][full_valid_idxs, m-1, i_lat, i_lon] = x1
        data_dict['sdhi1'][full_valid_idxs, m-1, i_lat, i_lon] = sdhi1
        data_dict['x2'][full_valid_idxs, m-1, i_lat, i_lon] = x2
        data_dict['sdhi2'][full_valid_idxs, m-1, i_lat, i_lon] = sdhi2

ds_monthly = xr.Dataset(
    {k: (['year', 'month', 'lat', 'lon'], v) for k, v in data_dict.items()},
    coords={'year': years, 'month': months, 'lat': lats, 'lon': lons}
)

OUTPUT_FILE = INDICES_DIR / "sdhi-monthly-ca-1951-2025.nc"
ds_monthly.to_netcdf(OUTPUT_FILE)
