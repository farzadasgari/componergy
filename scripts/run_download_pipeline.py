from __future__ import annotations

import argparse
import logging

from componergy.download.registry import download_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the componergy download pipeline."
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--skip-noaa", action="store_true", help="Skip NOAA nClimGrid downloads."
    )
    parser.add_argument("--noaa-start-year", type=int, default=1991)
    parser.add_argument("--noaa-end-year", type=int, default=2025)
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

    download_all(
        max_workers=args.max_workers,
        include_noaa=not args.skip_noaa,
        noaa_start_year=args.noaa_start_year,
        noaa_end_year=args.noaa_end_year,
    )


if __name__ == "__main__":
    main()
