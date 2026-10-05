"""CLI entry point for generating all supplementary figures"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate supplementary figures S1-S7.")
    parser.add_argument("--only", choices=["s1", "hovmoller", "sites", "all"], default="all",
                        help="Which group to build. 'hovmoller' is S2-S5; 'sites' is S6-S7.")
    parser.add_argument("--sapei-var", default="sapei_3m", help="Which SAPEI timescale column to use.")
    parser.add_argument("--scdhi-var", default="scdhi_3m", help="Which SCDHI timescale column to use.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.only in ("s1", "all"):
        from componergy.figures.cooccurrence import main as s1_main
        s1_main(scdhi_var=args.scdhi_var)

    if args.only in ("hovmoller", "all"):
        from componergy.figures.hovmoller import main as hov_main
        hov_main(sapei_var=args.sapei_var, scdhi_var=args.scdhi_var)

    if args.only in ("sites", "all"):
        from componergy.figures.sites import main as sites_main
        sites_main(sapei_var=args.sapei_var, scdhi_var=args.scdhi_var)


if __name__ == "__main__":
    main()
