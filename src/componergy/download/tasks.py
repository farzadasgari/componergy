"""Data structures describing a single downloadable dataset file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetTask:
    name: str
    source_page: str
    url: str
    destination: Path