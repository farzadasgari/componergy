"""
CLI entry point for aggregating California-cropped daily NOAA files into
one combined monthly-resolution climate dataset.
"""

from __future__ import annotations

import argparse
import logging

from componergy.preprocessing.aggregate_monthly import main as aggregate_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate cropped daily NOAA files to a monthly climate dataset."
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even if the output file already exists."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show INFO-level logs."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    aggregate_main(force=args.force)


if __name__ == "__main__":
    main()
