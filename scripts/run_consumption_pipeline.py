"""CLI entry point for ingesting the raw CEC county consumption workbook."""

from __future__ import annotations

import argparse

from componergy.preprocessing.consumption_cec import main as consumption_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process the CEC county consumption workbook into a clean monthly table.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the output file already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    consumption_main(force=args.force)


if __name__ == "__main__":
    main()
