"""Index-construction and county-to-region sensitivity checks (reviewer points on index design and spatial scale)."""

from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sapei-var", default="sapei_3m")
    p.add_argument("--scale", type=int, default=3, help="SPI accumulation months (match the SAPEI timescale)")
    return p.parse_args()
