import xarray as xr
import numpy as np
from scipy.stats import rankdata, norm, multivariate_normal
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def gringorten_cdf(arr):
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan)
    valid = ~np.isnan(arr)
    if valid.sum() < 3:
        return out
    ranks = rankdata(arr[valid], method="average")
    n = valid.sum()
    g = (ranks - 0.44) / (n + 0.12)
    g = np.clip(g, 1e-6, 1 - 1e-6)
    out[valid] = g
    return out


def standardized_index_from_series(arr):
    g = gringorten_cdf(arr)
    out = np.full_like(arr, np.nan)
    valid = ~np.isnan(g)
    out[valid] = norm.ppf(g[valid])
    return out


def joint_prob_dry_hot(spi_vals, sti_vals):
    spi = np.asarray(spi_vals, dtype=float)
    sti = np.asarray(sti_vals, dtype=float)
    p = np.full_like(spi, np.nan)
    valid = ~(np.isnan(spi) | np.isnan(sti))
    if valid.sum() < 3: return p
    u1 = spi[valid]
    u2 = sti[valid]
    rho = np.corrcoef(u1, u2)[0, 1]
    rho = np.clip(rho, -0.999, 0.999)
    cov = np.array([[1.0, rho],
                    [rho, 1.0]])
    mvn = multivariate_normal(mean=[0.0, 0.0], cov=cov)
    p_valid = np.zeros_like(u1)
    for i in range(len(u1)):
        u1_i = u1[i]
        u2_i = u2[i]
        p_u1 = norm.cdf(u1_i)
        p_u1_u2 = mvn.cdf([u1_i, u2_i])
        p_valid[i] = p_u1 - p_u1_u2
    p_valid = np.clip(p_valid, 1e-8, 1 - 1e-8)
    p[valid] = p_valid
    return p


def scei_from_spi_sti(spi_series, sti_series):
    p = joint_prob_dry_hot(spi_series, sti_series)
    g = gringorten_cdf(p)
    scei = np.full_like(p, np.nan)
    valid = ~np.isnan(g)
    if valid.sum() < 3:
        return scei
    scei[valid] = norm.ppf(g[valid])
    return scei


monthly_ds = xr.open_dataset("../dataset/monthly_ca_1951_2025.nc")
warm_da = xr.open_dataarray("../dataset/warm_seasons_ca.nc")

mask = ~np.isnan(warm_da).any(dim="window")
lat_idxs, lon_idxs = np.where(mask)

lats = monthly_ds.lat.values
lons = monthly_ds.lon.values

years = np.arange(1951, 2026)
n_years = len(years)

vars_to_save = ["P_warm", "Tmean_warm", "SPI", "STI", "SCEI"]
data_dict = {
    v: np.full((n_years, len(lats), len(lons)), np.nan, dtype=float)
    for v in vars_to_save
}

for c_idx in tqdm(range(len(lat_idxs)), desc="Cells"):
    i_lat = lat_idxs[c_idx]
    i_lon = lon_idxs[c_idx]

    lat_val = lats[i_lat]
    lon_val = lons[i_lon]

    warm_months = warm_da.isel(lat=i_lat, lon=i_lon).values.astype(int)
    cell = monthly_ds[["prcp", "tmean"]].sel(lat=lat_val, lon=lon_val, method="nearest")

    P_warm = np.full(n_years, np.nan)
    T_warm = np.full(n_years, np.nan)

    for y_idx, year in enumerate(years):
        ydata = cell.sel(time=cell["time.year"] == year)
        if ydata.time.size == 0:
            continue

        months = ydata["time.month"].values
        m_mask = np.isin(months, warm_months)
        if m_mask.sum() != 3: continue
        P_vals = ydata["prcp"].values[m_mask]
        T_vals = ydata["tmean"].values[m_mask]

        P_warm[y_idx] = P_vals.sum()
        T_warm[y_idx] = T_vals.mean()

    valid_warm = ~(np.isnan(P_warm) | np.isnan(T_warm))
    if valid_warm.sum() < 10: continue
    P_valid = P_warm[valid_warm]
    T_valid = T_warm[valid_warm]

    SPI_valid = standardized_index_from_series(P_valid)
    STI_valid = standardized_index_from_series(T_valid)

    SPI = np.full_like(P_warm, np.nan)
    STI = np.full_like(T_warm, np.nan)

    full_valid_idxs = np.where(valid_warm)[0]
    SPI[full_valid_idxs] = SPI_valid
    STI[full_valid_idxs] = STI_valid
    SCEI = scei_from_spi_sti(SPI, STI)

    for y_idx in range(n_years):
        if np.isnan(P_warm[y_idx]) or np.isnan(T_warm[y_idx]):
            continue
        data_dict["P_warm"][y_idx, i_lat, i_lon] = P_warm[y_idx]
        data_dict["Tmean_warm"][y_idx, i_lat, i_lon] = T_warm[y_idx]
        data_dict["SPI"][y_idx, i_lat, i_lon] = SPI[y_idx]
        data_dict["STI"][y_idx, i_lat, i_lon] = STI[y_idx]
        data_dict["SCEI"][y_idx, i_lat, i_lon] = SCEI[y_idx]

ds_scei = xr.Dataset(
    {name: (["year", "lat", "lon"], arr) for name, arr in data_dict.items()},
    coords={
        "year": years,
        "lat": lats,
        "lon": lons,
    }
)

ds_scei.attrs["description"] = (
    "Warm-season precipitation and temperature, SPI, STI, "
    "and Standardized Compound Event Indicator (SCEI) "
)
out_path = "../dataset/ca_scei_1951_2025.nc"
ds_scei.to_netcdf(out_path)
