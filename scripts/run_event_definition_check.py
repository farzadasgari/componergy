"""Rank event definitions by how closely they reproduce the manuscript's quoted event counts."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sapei-var", default="sapei_3m")
    p.add_argument("--scdhi-var", default="scdhi_3m")
    return p.parse_args()
