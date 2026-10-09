"""Region abbreviations"""

from __future__ import annotations

import pandas as pd

REGION_ABBREV = {
    "San Joaquin Valley": "SJV",
    "Sacramento-Delta": "SD",
    "Central Coast": "CC",
    "North Coast": "NC",
    "South Coast": "SC",
    "North Central": "N-C",
    "Southern Interior": "SI",
    "Mojave Desert": "MD",
    "Sonoran Desert": "SoD",
    "Northeast": "NE",
    "Sierra": "SR",
}


def _initials(name: str, width: int = 3) -> str:
    parts = [p for p in name.replace("-", " ").split() if p]
    return "".join(p[0].upper() for p in parts[:width])


def region_abbrev(name: str) -> str:
    """The manuscript abbreviation, or initials for a name that has none."""
    return REGION_ABBREV.get(name, _initials(name))
