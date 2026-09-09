"""
CLI entry point for cropping raw NOAA nClimGrid-Daily files to California.

Kept as a tracked script (not a notebook).
"""

from __future__ import annotations

import argparse
import logging

from componergy.preprocessing.crop_nclimgrid import main as crop_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop raw NOAA nClimGrid-Daily files to California."
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
    crop_main()


if __name__ == "__main__":
    main()