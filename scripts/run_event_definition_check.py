"""Rank event definitions by how closely they reproduce the manuscript's quoted event counts."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sapei-var", default="sapei_3m")
    p.add_argument("--scdhi-var", default="scdhi_3m")
    return p.parse_args()


def main() -> None:
    import pandas as pd
    import xarray as xr

    from componergy.analysis.event_definition_check import event_definition_check
    from componergy.figures.composite_stats import zscore
    from componergy.figures.trend_stats import compute_area_weights, compute_statewide_mean
    from componergy.paths import FIGURES_DIR, INDICES_FILE

    args = parse_args()
    ds = xr.open_dataset(INDICES_FILE)
    w = compute_area_weights(ds["lat"])
    series = {k: zscore(compute_statewide_mean(ds[v], w).to_series()) for k, v in
              (("sti", "sti"), ("sapei", args.sapei_var), ("scdhi", args.scdhi_var))}
    table = event_definition_check(series["sti"], series["sapei"], series["scdhi"])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "event_definition_check.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table[table["Best_for_window"]].to_string(index=False))
    print(f"\nfull table ({len(table)} rows) written to {out}")


if __name__ == "__main__":
    main()
