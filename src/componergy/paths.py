"""Centralized filesystem paths for the componergy project."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_CLIMATE_DIR = RAW_DATA_DIR / "climate"
CA_BOUNDARY_DIR = RAW_DATA_DIR / "boundaries" / "california"

MANIFEST_PATH = DATA_DIR / "download_manifest.json"


def ensure_dirs() -> None:
    """Create all pipeline directories if they don't already exist."""
    for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_CLIMATE_DIR, CA_BOUNDARY_DIR):
        d.mkdir(parents=True, exist_ok=True)