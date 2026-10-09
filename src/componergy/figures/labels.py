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


def duplicate_labels(names) -> list:
    """Labels used by more than one of `names` (empty when the set is unambiguous)."""
    labels = [region_abbrev(n) for n in names]
    return sorted({lab for lab in labels if labels.count(lab) > 1})


def region_label_table(names) -> pd.DataFrame:
    """Region -> abbreviation, with provenance, guaranteed unique."""
    names = list(names)
    manuscript = [n for n in names if n in REGION_ABBREV]
    dup_m = {REGION_ABBREV[n] for n in manuscript if [REGION_ABBREV[m] for m in manuscript].count(REGION_ABBREV[n]) > 1}
    if dup_m:
        raise ValueError(f"REGION_ABBREV assigns the same label to different regions: {sorted(dup_m)}")

    rows, used = [], {REGION_ABBREV[n] for n in manuscript}
    for n in names:
        if n in REGION_ABBREV:
            rows.append({"Region": n, "Abbreviation": REGION_ABBREV[n], "Source": "manuscript"})
            continue
        lab, width = _initials(n), 3
        while lab in used and width < len(n):
            width += 1
            lab = _initials(n, width) if width <= len(n.split()) else (_initials(n) + str(width))
        used.add(lab)
        rows.append({"Region": n, "Abbreviation": lab, "Source": "fallback (no manuscript label)"})
    return pd.DataFrame(rows)
