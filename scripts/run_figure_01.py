# scripts/run_figure_01.py
"""CLI entry point for generating Figure 1."""

from __future__ import annotations

import argparse

from componergy.figures.figure_01_trend import main as figure_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 1 (statewide signals and trends).")
    parser.add_argument("--sapei-var", default="sapei_3m", help="Which SAPEI timescale column to use.")
    parser.add_argument("--scdhi-var", default="scdhi_3m", help="Which SCDHI timescale column to use.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure_main(sapei_var=args.sapei_var, scdhi_var=args.scdhi_var)


if __name__ == "__main__":
    main()