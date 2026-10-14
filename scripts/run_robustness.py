"""Compound effect and amplification contrast across event baseline x smoothing, for every outcome family."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sapei-var", default="sapei_3m")
    p.add_argument("--scdhi-var", default="scdhi_3m")
    p.add_argument("--force", action="store_true", help="Rebuild even if the output files already exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from componergy.paths import FIGURES_DIR

    out_dir = FIGURES_DIR / "robustness"
    csv_path, log_path = out_dir / "robustness_long.csv", out_dir / "robustness.log"
    if csv_path.exists() and log_path.exists() and not args.force:
        print(f"{csv_path} and {log_path} exist; skipping (use --force to rebuild)")
        return

    import pandas as pd
    import xarray as xr

    from componergy.analysis.regions import build_county_to_region_map
    from componergy.analysis.robustness import (
        BASELINES, SMOOTHINGS, FAMILY_TRENDS, MDE80_FACTOR, COLUMNS, family_rows, shift_summary,
        consistency_section,
    )
    from componergy.download.boundaries import get_california_counties
    from componergy.download.climate_regions import get_california_climate_regions
    from componergy.figures.generation import prepare_data as prepare_generation_data
    from componergy.figures.imbalance import prepare_imbalance_data, load_demand_sources
    from componergy.figures.zonal_consumption import prepare_zonal_data
    from componergy.netcdf_io import atomic_to_csv
    from componergy.paths import INDICES_FILE, GENERATION_MONTHLY_FILE, COUNTY_CONSUMPTION_XLSX
    from componergy.preprocessing.consumption_cec import load_raw_consumption

    print("loading climate indices, generation, CEC consumption and regions...")
    indices_ds = xr.open_dataset(INDICES_FILE)
    generation_df = pd.read_csv(GENERATION_MONTHLY_FILE, parse_dates=[0], index_col=0)
    generation_df.index = pd.DatetimeIndex(generation_df.index.values, name=None)
    cec_demand = load_demand_sources()["cec"]
    if cec_demand is None:
        raise FileNotFoundError(f"the CEC county consumption workbook is required: {COUNTY_CONSUMPTION_XLSX}")
    regions_gdf = get_california_climate_regions()
    county_region_map = build_county_to_region_map(get_california_counties(), regions_gdf)
    consumption_long_df = load_raw_consumption(COUNTY_CONSUMPTION_XLSX)
    var_kw = {"sapei_var": args.sapei_var, "scdhi_var": args.scdhi_var}

    rows, shifts, periods = [], [], {}
    for baseline in BASELINES:
        for smoothing in SMOOTHINGS:
            print(f"baseline={baseline} smoothing={smoothing}...")
            kw = {**var_kw, "smooth_months": smoothing, "baseline": baseline}
            imb = prepare_imbalance_data(indices_ds, generation_df, cec_demand.to_frame("total_sales"), **kw)
            zonal = prepare_zonal_data(indices_ds, consumption_long_df, regions_gdf, county_region_map, **kw)
            gen = prepare_generation_data(indices_ds, generation_df, **kw)
            for family, raw_df, d in (("imbalance", imb["imb_gwh"].to_frame("Imbalance"), imb),
                                      ("regional_demand", zonal["electricity_raw"], zonal),
                                      ("generation_share", gen["mix_pct_sm"], gen)):
                r, s = family_rows(raw_df, d["event_masks"], d["normal"], d["event_meta"], family,
                                   baseline, smoothing, FAMILY_TRENDS[family])
                rows.append(r)
                shifts.append(s)
                periods[family] = f"{raw_df.index.min().date()} to {raw_df.index.max().date()}"

    table = pd.concat(rows, ignore_index=True)[list(COLUMNS)]
    summary = shift_summary(pd.concat(shifts, ignore_index=True))
    per_baseline = summary.groupby("baseline", sort=False).agg(S_min=("S_min", "min"), S_max=("S_max", "max"))
    per_baseline["p_floor_at_S_min"] = 1.0 / (per_baseline["S_min"] + 1)
    per_baseline["p_floor_at_S_max"] = 1.0 / (per_baseline["S_max"] + 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_to_csv(table, csv_path, index=False)
    consistency, counts = consistency_section(table, FIGURES_DIR)
    lines = [
        "Robustness: compound effect and amplification contrast over baseline x smoothing",
        f"grid: baseline {BASELINES} x smoothing {SMOOTHINGS} months; events 90/10/10; month FE + trend;",
        "trend per family, the one its main figure uses: "
        + ", ".join(f"{k} {v}" for k, v in FAMILY_TRENDS.items()) + ";",
        "inference: calendar-shift randomization; BH-FDR within family.",
        "",
        "periods: " + "; ".join(f"{k} {v}" for k, v in periods.items()),
        "",
        "columns:",
        "  n_heat, n_drought, n_compound: months meeting each univariate criterion in the analysis window",
        "  n_compound_exclusive: compound months entering the fit for this outcome",
        "  effect_pct, effect_z: compound effect, % of the fitted no-event baseline and in anomaly-SD units",
        "  p_exact: exact randomization p of the compound effect (floor 1/(S+1))",
        "  q_bh: BH over all outcome x event tests of the family, on the studentized randomization p "
        "(the convention of the figure CSVs)",
        "  A, A_ci_lo, A_ci_hi, SE_A: compound - (heat-only + drought-only), anomaly-SD units, 95% CI",
        "  A_p: studentized randomization p of A; A_q: BH across the family's outcomes",
        f"  MDE80 = {MDE80_FACTOR} * SE_A (two-sided alpha 0.05, power 0.80)",
        "  NaN = not defined (e.g. fewer than 3 event months), never filled",
        "  trend: trend term of the month-FE fit for this row (the TREND constant of the family's main figure: "
        "figure_02 generation, figure_04 zonal_consumption, figure_06 imbalance)",
        "",
        "shift counts S and p-value floor 1/(S+1), per baseline:",
        per_baseline.to_string(),
        "",
        "by baseline, smoothing and family:",
        summary.to_string(index=False),
        "",
        "CONSISTENCY WITH MAIN FIGURES",
        *consistency,
        f"summary: {counts['OK']} OK, {counts['MISMATCH']} MISMATCH, {counts['SKIPPED']} SKIPPED",
        "",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {csv_path} ({len(table)} rows) and {log_path}")
    print(f"consistency with main figures: {counts['OK']} OK, {counts['MISMATCH']} MISMATCH, "
          f"{counts['SKIPPED']} SKIPPED")


if __name__ == "__main__":
    main()
