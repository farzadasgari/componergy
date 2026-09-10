from __future__ import annotations

import xarray as xr

from componergy.indices.pet import add_pet
from componergy.indices.sti import add_sti
from componergy.indices.sapei import add_sapei
from componergy.indices.scdhi import add_scdhi

from componergy.paths import NOAA_MONTHLY_FILE, INDICES_FILE, ensure_dirs


def build_all_indices(monthly_ds: xr.Dataset, n_jobs: int = -1, copula_family: str = None) -> xr.Dataset:
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
