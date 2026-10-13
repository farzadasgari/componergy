"""
Monthly potential evapotranspiration (PET) via the Hargreaves-Samani
equation, applied to monthly-mean temperature.

Units: temperatures in degC; extraterrestrial radiation Ra is computed in
MJ m^-2 day^-1 and converted to mm/day of evaporation equivalent
(x 0.408, FAO-56 Eq. 20) before entering Hargreaves-Samani (FAO-56 Eq. 52),
so PET comes out in mm/day and is returned as a monthly total in mm/month.
"""

from __future__ import annotations

import calendar
from math import cos, pi, radians, sin

import numpy as np
import pandas as pd
import xarray as xr

GSC = 0.0820  # MJ m^-2 min^-1, FAO-56 solar constant
HARGREAVES_COEFF = 0.0023
HARGREAVES_TEMP_OFFSET = 17.8
# MJ m^-2 day^-1 -> mm/day evaporation equivalent (1 / latent heat of
# vaporization, 2.45 MJ/kg); FAO-56, Allen et al. (1998), Eq. 20 and Eq. 52.
RA_MJ_TO_MM_PER_DAY = 0.408


def _extraterrestrial_radiation(lat_deg: float, day_of_year: float) -> float:
    """Ra (extraterrestrial radiation, MJ/m^2/day), standard FAO-56 formula."""
    lat_rad = radians(lat_deg)
    dr = 1 + 0.033 * cos(2 * pi * day_of_year / 365)
    sol_dec = 0.409 * sin(2 * pi * day_of_year / 365 - 1.39)
    sunset_hour_angle = np.arccos(
        np.clip(-np.tan(lat_rad) * np.tan(sol_dec), -1.0, 1.0)
    )
    return (
            (24 * 60 / pi)
            * GSC
            * dr
            * (
                    sunset_hour_angle * sin(lat_rad) * sin(sol_dec)
                    + cos(lat_rad) * cos(sol_dec) * np.sin(sunset_hour_angle)
            )
    )


def _ra_by_month_lat(lats: np.ndarray) -> np.ndarray:
    """Ra (MJ m^-2 day^-1) for every (month, lat) combination. Shape: (12, n_lat)."""
    ra = np.zeros((12, len(lats)))
    for month in range(1, 13):
        mid_month_doy = 30.44 * (month - 1) + 15
        for i, lat in enumerate(lats):
            ra[month - 1, i] = _extraterrestrial_radiation(lat, mid_month_doy)
    return ra


def hargreaves_pet_monthly(lat_deg: float, month: int, tavg: float, tmax: float, tmin: float,
                           days_in_month: int) -> float:
    """
    Monthly total PET (mm/month) for a single (lat, month, year) value.

    Temperatures are in degC. Ra (MJ m^-2 day^-1) is converted to mm/day
    with RA_MJ_TO_MM_PER_DAY before applying Hargreaves-Samani, giving
    mm/day, which is multiplied by `days_in_month`.
    """
    if np.isnan(tavg) or np.isnan(tmax) or np.isnan(tmin):
        return np.nan

    mid_month_doy = 30.44 * (month - 1) + 15
    ra_mm_per_day = RA_MJ_TO_MM_PER_DAY * _extraterrestrial_radiation(lat_deg, mid_month_doy)
    trange = max(tmax - tmin, 0.0)
    pet_daily_equivalent = max(
        HARGREAVES_COEFF * ra_mm_per_day * (tavg + HARGREAVES_TEMP_OFFSET) * np.sqrt(trange), 0.0
    )
    return pet_daily_equivalent * days_in_month


def add_pet(monthly_ds: xr.Dataset) -> xr.Dataset:
    """
    Add a 'pet' variable (mm/month) to a monthly climate dataset.

    Vectorised equivalent of hargreaves_pet_monthly: tavg/tmax/tmin in degC,
    Ra converted from MJ m^-2 day^-1 to mm/day with RA_MJ_TO_MM_PER_DAY.
    """
    lats = monthly_ds["lat"].values
    times = pd.DatetimeIndex(monthly_ds["time"].values)
    months = times.month
    days_in_month = np.array([calendar.monthrange(t.year, t.month)[1] for t in times])

    ra_lookup = RA_MJ_TO_MM_PER_DAY * _ra_by_month_lat(lats)  # (12, n_lat), mm/day
    ra_per_time = ra_lookup[months - 1, :]  # (time, n_lat)
    ra_per_time = ra_per_time[:, :, None]  # (time, n_lat, 1) -- broadcast over lon

    tavg = monthly_ds["tavg"].values
    tmax = monthly_ds["tmax"].values
    tmin = monthly_ds["tmin"].values
    trange = np.clip(tmax - tmin, 0, None)

    pet_daily_equiv = np.clip(
        HARGREAVES_COEFF * ra_per_time * (tavg + HARGREAVES_TEMP_OFFSET) * np.sqrt(trange),
        0, None,
    )
    pet_monthly = pet_daily_equiv * days_in_month[:, None, None]

    nan_mask = np.isnan(tavg) | np.isnan(tmax) | np.isnan(tmin)
    pet_monthly = np.where(nan_mask, np.nan, pet_monthly)

    out = monthly_ds.copy()
    out["pet"] = (("time", "lat", "lon"), pet_monthly)
    out["pet"].attrs["units"] = "mm/month"
    out["pet"].attrs["long_name"] = "Potential evapotranspiration (Hargreaves-Samani, monthly total)"
    return out
