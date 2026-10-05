"""CLI entry point for scanning cropped NOAA files for corruption."""

from __future__ import annotations

import argparse

from componergy.preprocessing.verify_crop import main as verify_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan cropped NOAA files for corruption.")
    parser.add_argument("--delete", action="store_true", help="Delete corrupted files found.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_main(delete=args.delete)


if __name__ == "__main__":
    main()
