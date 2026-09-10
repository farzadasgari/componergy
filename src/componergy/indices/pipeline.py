"""
End-to-end index construction: climate.nc -> +PET -> +STI -> +SAPEI -> +SCDHI.

Wires together the four index modules in the order each depends on the
last (SAPEI needs PET; SCDHI needs both SAPEI and STI), against the
single combined monthly climate dataset, and writes one final dataset containing
every month from the full record with all indices attached.
"""

from __future__ import annotations

import xarray as xr

from componergy.indices.pet import add_pet
from componergy.indices.sti import add_sti
from componergy.indices.sapei import add_sapei
from componergy.indices.scdhi import add_scdhi
from componergy.paths import NOAA_MONTHLY_FILE, INDICES_FILE, ensure_dirs


def build_all_indices(monthly_ds: xr.Dataset, n_jobs: int = -1, copula_family: str = None) -> xr.Dataset:
    """
    Add PET, STI, SAPEI (all 4 timescales), and SCDHI (all 4 timescales)
    to a monthly climate dataset, in dependency order.

    Parameters
    ----------
    monthly_ds : xr.Dataset
        Must contain tavg, tmax, tmin, prcp.
    n_jobs : int
        Passed through to add_sapei/add_scdhi's per-cell parallel fitting.
    copula_family : str, optional
        Fix the SCDHI copula family instead of auto-selecting it.

    Returns
    -------
    xr.Dataset
        The input dataset with all index variables added.
    """
    print("computing PET...")
    ds = add_pet(monthly_ds)
    print("PET done.")

    print("computing STI...")
    ds = add_sti(ds)
    print("STI done.")

    print("computing SAPEI (3/6/9/12-month)...")
    ds = add_sapei(ds, n_jobs=n_jobs)
    print("SAPEI done.")

    print("computing SCDHI (3/6/9/12-month)...")
    ds = add_scdhi(ds, family_name=copula_family, n_jobs=n_jobs)
    print("SCDHI done.")

    return ds


def main(n_jobs: int = -1, copula_family: str = None, force: bool = False) -> None:
    """
    Build all indices from NOAA_MONTHLY_FILE and write INDICES_FILE.

    Idempotent: skips entirely if INDICES_FILE already exists, unless
    force=True.
    """
    ensure_dirs()

    if INDICES_FILE.exists() and not force:
        print(f"{INDICES_FILE} already exists, skipping (pass force=True to rebuild).")
        return

    print(f"loading {NOAA_MONTHLY_FILE}...")
    monthly_ds = xr.open_dataset(NOAA_MONTHLY_FILE)

    result = build_all_indices(monthly_ds, n_jobs=n_jobs, copula_family=copula_family)

    print(f"writing {INDICES_FILE}...")
    encoding = {v: {"zlib": True, "complevel": 5} for v in result.data_vars}
    result.to_netcdf(INDICES_FILE, encoding=encoding)
    print(f"wrote {INDICES_FILE}: {dict(result.sizes)}")
