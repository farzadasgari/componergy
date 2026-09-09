"""Data structures describing a single downloadable dataset file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetTask:
    """A single file to download as part of the data-acquisition pipeline.

    Attributes
    ----------
    name : str
        Human-readable name, used in logs and the download manifest.
    source_page : str
        URL of the page documenting/describing the dataset (for provenance,
        not used for the download itself).
    url : str
        Direct download URL for this file.
    destination : Path
        Local path the file should be saved to.
    """
    name: str
    source_page: str
    url: str
    destination: Path
