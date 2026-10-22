"""Write 100 dpi JPEG previews of every figure under figures/ into figures/previews/ (sources are never modified)."""

from __future__ import annotations

import argparse

from componergy.figures.previews import main as previews_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rewrite previews even if they are newer than their source.")
    return parser.parse_args()


def main() -> None:
    previews_main(force=parse_args().force)


if __name__ == "__main__":
    main()
