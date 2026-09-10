"""
CLI entry point for building PET, STI, SAPEI, and SCDHI from the combined
monthly climate dataset.
"""

from __future__ import annotations

import argparse
import logging

from componergy.indices.pipeline import main as pipeline_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PET, STI, SAPEI, and SCDHI from the monthly climate dataset."
    )
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel workers for SAPEI/SCDHI fitting (-1 = all cores).")
    parser.add_argument("--copula-family", default=None, choices=["frank", "gaussian", "clayton", "gumbel"],
                        help="Fix the SCDHI copula family instead of auto-selecting it.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the output file already exists.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show INFO-level logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    pipeline_main(n_jobs=args.n_jobs, copula_family=args.copula_family, force=args.force)


if __name__ == "__main__":
    main()
