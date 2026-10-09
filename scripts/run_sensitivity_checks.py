"""Index-construction and county-to-region sensitivity checks (reviewer points on index design and spatial scale)."""

from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sapei-var", default="sapei_3m")
    p.add_argument("--scale", type=int, default=3, help="SPI accumulation months (match the SAPEI timescale)")
    return p.parse_args()


def main() -> None:
    import pandas as pd
    import xarray as xr

    from componergy import paths
    from componergy.analysis.index_diagnostics import (
        standardized_precip_index, temperature_dependence, drought_definition_agreement,
    )
    from componergy.figures.composite_stats import zscore
    from componergy.figures.trend_stats import compute_area_weights, compute_statewide_mean

    args = parse_args()
    out = paths.FIGURES_DIR / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(paths.INDICES_FILE)
    if "prcp" not in ds and paths.NOAA_MONTHLY_FILE.exists():
        ds = ds.merge(xr.open_dataset(paths.NOAA_MONTHLY_FILE)[["prcp"]], compat="override")
    w = compute_area_weights(ds["lat"])
    sti = zscore(compute_statewide_mean(ds["sti"], w).to_series())
    sapei = zscore(compute_statewide_mean(ds[args.sapei_var], w).to_series())
    spi = standardized_precip_index(compute_statewide_mean(ds["prcp"], w).to_series(), scale=args.scale)

    dep = temperature_dependence(sti, sapei, spi)
    (out / "index_temperature_dependence.json").write_text(json.dumps(dep, indent=2))
    rows = []
    for label, window in (("full record", None),
                          ("2008-2024", sti.index[(sti.index >= "2008-01-01") & (sti.index <= "2024-12-01")]),
                          ("2001-2025", sti.index[(sti.index >= "2001-01-01") & (sti.index <= "2025-08-01")])):
        rows.append({"Window": label, **drought_definition_agreement(sapei, spi, window=window)})
    pd.DataFrame(rows).to_csv(out / "drought_definition_agreement.csv", index=False)
    print("temperature dependence:", dep["verdict"])

    from componergy.analysis.regions import county_region_shares, county_split_summary, allocation_sensitivity
    from componergy.download.boundaries import get_california_counties
    from componergy.download.climate_regions import get_california_climate_regions

    shares = county_region_shares(get_california_counties(), get_california_climate_regions())
    split = county_split_summary(shares)
    split.to_csv(out / "county_region_split.csv", index=False)
    print(f"counties with majority share < 0.8: {int(split['Split'].sum())} of {len(split)}")

    if paths.COUNTY_CONSUMPTION_XLSX.exists():
        from componergy.preprocessing.consumption_cec import load_raw_consumption
        alloc = allocation_sensitivity(load_raw_consumption(paths.COUNTY_CONSUMPTION_XLSX), shares)
        alloc["per_region"].assign(share_of_consumption_changing_region=alloc["share_of_consumption_changing_region"]
                                   ).to_csv(out / "county_allocation_sensitivity.csv", index=False)
        print(
            f"share of statewide consumption that changes region: {alloc['share_of_consumption_changing_region']:.3f}")
    else:
        print("CEC workbook not present: allocation sensitivity skipped")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
