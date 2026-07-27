import xarray as xr
import numpy as np
from scipy.stats import rankdata, norm
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

monthly_ds = xr.open_dataset("../dataset/monthly_ca_1951_2025.nc")
warm_da = xr.open_dataarray("../dataset/warm_seasons_ca.nc")

mask = ~np.isnan(warm_da).any(dim='window')
lat_idxs, lon_idxs = np.where(mask)
n_cells = len(lat_idxs)
years = np.arange(1951, 2026)
n_years = len(years)

lats = monthly_ds.lat.values
lons = monthly_ds.lon.values

vars_to_save = [
    'prcp', 'tmean', 'tmax', 'tmin', 'pet',
    'spi', 'sti', 'spei',
    'x1', 'sdhi1', 'x2', 'sdhi2'
]
data_dict = {v: np.full((n_years, len(lats), len(lons)), np.nan) for v in vars_to_save}

print(f"Processing {n_cells} valid cells → building ultimate dataset...")

for c_idx in tqdm(range(n_cells), desc="Cells"):
    i_lat = lat_idxs[c_idx]
    i_lon = lon_idxs[c_idx]
    lat = lats[i_lat]
    lon = lons[i_lon]

    window = warm_da.isel(lat=i_lat, lon=i_lon).values.astype(int)
    cell = monthly_ds[['prcp', 'tmean', 'tmax', 'tmin']].sel(lat=lat, lon=lon, method='nearest')

    P, Tmean, Tmax, Tmin, PET = [], [], [], [], []

    for year in years:
        ydata = cell.sel(time=cell['time.year'] == year)
        months = ydata['time.month'].values
        mask = np.isin(months, window)

        if mask.sum() != 3:
            P.append(np.nan);
            Tmean.append(np.nan);
            Tmax.append(np.nan);
            Tmin.append(np.nan);
            PET.append(np.nan)
            continue

        p_vals = ydata['prcp'].values[mask]
        t_vals = ydata['tmean'].values[mask]
        tx_vals = ydata['tmax'].values[mask]
        tn_vals = ydata['tmin'].values[mask]

        P.append(p_vals.mean())
        Tmean.append(t_vals.mean())
        Tmax.append(tx_vals.mean())
        Tmin.append(tn_vals.mean())

        doy = [np.mean([31 * (m - 1) + 15 for m in window])]
        from math import sin, radians, pi

        lat_rad = radians(lat)
        pet_daily = []
        for m in window:
            sol_dec = 0.409 * sin(2 * pi * (doy[0]) / 365 - 1.39)
            sha = np.arccos(-np.tan(lat_rad) * np.tan(sol_dec))
            ra = 15.39 * (24 / pi) * sha * (0.0835 * sin(2 * pi * doy[0] / 365 - 1.39) + 0.0007)
            trange = tx_vals[window.tolist().index(m)] - tn_vals[window.tolist().index(m)]
            pet_d = 0.0023 * ra * (t_vals[window.tolist().index(m)] + 17.8) * np.sqrt(max(trange, 0))
            pet_daily.append(pet_d)
        PET.append(np.mean(pet_daily) * 30.4)

    P = np.array(P)
    Tmean = np.array(Tmean)
    Tmax = np.array(Tmax)
    Tmin = np.array(Tmin)
    PET = np.array(PET)
    valid = ~(np.isnan(P) | np.isnan(Tmean))

    if valid.sum() < 10:
        continue


    def std_index(arr):
        ranks = rankdata(arr[valid], method='average')
        n = ranks.size
        p = (ranks - 0.44) / (n + 0.12)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return norm.ppf(p)


    spi = std_index(P)
    sti = std_index(Tmean)
    spei = std_index(P - PET)

    g1_p = (rankdata(P[valid], 'average') - 0.44) / (valid.sum() + 0.12)
    g2_t = (rankdata(Tmean[valid], 'average') - 0.44) / (valid.sum() + 0.12)
    g1_pe = (rankdata((P - PET)[valid], 'average') - 0.44) / (valid.sum() + 0.12)

    g1_p = np.clip(g1_p, 1e-6, 1 - 1e-6)
    g2_t = np.clip(g2_t, 1e-6, 1 - 1e-6)
    g1_pe = np.clip(g1_pe, 1e-6, 1 - 1e-6)

    x1 = g1_p / g2_t
    x2 = g1_pe / g2_t

    sdhi1 = norm.ppf((rankdata(x1, 'average') - 0.44) / (len(x1) + 0.12))
    sdhi2 = norm.ppf((rankdata(x2, 'average') - 0.44) / (len(x2) + 0.12))

    for y_idx, year in enumerate(years):
        if not valid[y_idx]: continue
        v_idx = np.where(valid)[0][np.where(years[valid] == year)[0][0]]
        data_dict['prcp'][y_idx, i_lat, i_lon] = P[y_idx]
        data_dict['tmean'][y_idx, i_lat, i_lon] = Tmean[y_idx]
        data_dict['tmax'][y_idx, i_lat, i_lon] = Tmax[y_idx]
        data_dict['tmin'][y_idx, i_lat, i_lon] = Tmin[y_idx]
        data_dict['pet'][y_idx, i_lat, i_lon] = PET[y_idx]
        data_dict['spi'][y_idx, i_lat, i_lon] = spi[v_idx]
        data_dict['sti'][y_idx, i_lat, i_lon] = sti[v_idx]
        data_dict['spei'][y_idx, i_lat, i_lon] = spei[v_idx]
        data_dict['x1'][y_idx, i_lat, i_lon] = x1[v_idx]
        data_dict['sdhi1'][y_idx, i_lat, i_lon] = sdhi1[v_idx]
        data_dict['x2'][y_idx, i_lat, i_lon] = x2[v_idx]
        data_dict['sdhi2'][y_idx, i_lat, i_lon] = sdhi2[v_idx]

ds_final = xr.Dataset(
    {k: (['year', 'lat', 'lon'], v) for k, v in data_dict.items()},
    coords={'year': years, 'lat': lats, 'lon': lons}
)

ds_final.attrs['title'] = 'California Warm-Season Compound Climate Extremes 1951-2025'
ds_final.attrs['source'] = 'nClimGrid + Hargreaves PET + Gringorten ranking'

ds_final.to_netcdf("../dataset/ca_sdhi_1951_2025.nc")
