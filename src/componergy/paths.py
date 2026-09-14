"""Centralized filesystem paths for the componergy project."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_CLIMATE_DIR = RAW_DATA_DIR / "climate"
CA_BOUNDARY_DIR = RAW_DATA_DIR / "boundaries" / "california"

# California-cropped copies of the raw NOAA nClimGrid-Daily files
NOAA_CA_DIR = INTERMEDIATE_DATA_DIR / "nclimgrid-ca"

# Single combined monthly-resolution climate dataset spanning the full
# record (all cropped daily files aggregated to one file per month, one
# time dimension)
NOAA_MONTHLY_FILE = PROCESSED_DATA_DIR / "california_monthly_climate.nc"

# Final combined dataset: climate + PET + STI + SAPEI + SCDHI, all months
# from the full record
INDICES_FILE = PROCESSED_DATA_DIR / "california_monthly_indices.nc"

MANIFEST_PATH = DATA_DIR / "download_manifest.json"

# Output directory for generated figures
FIGURES_DIR = ROOT / "figures"

RAW_DATA_ELECTRICITY_DIR = RAW_DATA_DIR / "electricity"

# Raw EIA-923 "generation_monthly.xlsx" (multi-sheet, all states)
EIA_GENERATION_RAW_FILE = RAW_DATA_ELECTRICITY_DIR / "generation_monthly.xlsx"

# Processed California-only, source-grouped monthly generation table
GENERATION_MONTHLY_FILE = PROCESSED_DATA_DIR / "california_generation_monthly.csv"

# Directories the pipeline needs to exist
_PIPELINE_DIRS = (
    RAW_DATA_DIR,
    INTERMEDIATE_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_CLIMATE_DIR,
    CA_BOUNDARY_DIR,
    NOAA_CA_DIR,
    FIGURES_DIR,
)


def ensure_dirs() -> None:
    """Create all pipeline directories if they don't already exist."""
    for d in _PIPELINE_DIRS:
        d.mkdir(parents=True, exist_ok=True)
