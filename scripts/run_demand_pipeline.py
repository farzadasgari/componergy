"""CLI entry point for ingesting the raw EIA-861M demand workbooks."""

from __future__ import annotations

import argparse

from componergy.preprocessing.demand_eia861m import main as demand_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process EIA-861M workbooks into a combined statewide demand table.")
    parser.add_argument("--state", default="CA", help="Two-letter state code to filter to.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the output file already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demand_main(state=args.state, force=args.force)


if __name__ == "__main__":
    main()
