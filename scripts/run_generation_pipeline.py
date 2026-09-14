"""CLI entry point for ingesting the raw EIA generation file into a California monthly table."""

from __future__ import annotations

import argparse

from componergy.preprocessing.generation_eia import main as generation_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process EIA generation_monthly.xlsx into a California monthly table.")
    parser.add_argument("--state", default="CA", help="Two-letter state code to filter to.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the output file already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_main(state=args.state, force=args.force)


if __name__ == "__main__":
    main()
