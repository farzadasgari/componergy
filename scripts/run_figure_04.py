"""CLI entry point for generating Figure 4 (zonal consumption response to climate events)."""

from __future__ import annotations

import argparse

from componergy.figures.zonal_consumption import main as figure_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 4 (zonal consumption response maps and bubble matrix).")
    parser.add_argument("--sapei-var", default="sapei_3m", help="Which SAPEI timescale column to use.")
    parser.add_argument("--scdhi-var", default="scdhi_3m", help="Which SCDHI timescale column to use.")
    parser.add_argument("--baseline", choices=["full_record", "window"], default="full_record",
                        help="Where event percentile thresholds are computed. \"full_record\" (default): "
                             "thresholds come from the whole 1951-2026 record, so an event means the same "
                             "thing in every figure and study window. \"window\": recomputes thresholds "
                             "inside each analysis window, which forces ~10%% of months to be events by "
                             "construction; kept only as a sensitivity option. See analysis.events.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure_main(sapei_var=args.sapei_var, scdhi_var=args.scdhi_var, baseline=args.baseline)


if __name__ == "__main__":
    main()
