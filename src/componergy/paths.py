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
CA_CLIMATE_REGIONS_DIR = RAW_DATA_DIR / "boundaries" / "california_climate_regions"
CA_COUNTIES_DIR = RAW_DATA_DIR / "boundaries" / "california_counties"

# California-cropped copies of the raw NOAA nClimGrid-Daily files
NOAA_CA_DIR = INTERMEDIATE_DATA_DIR / "nclimgrid-ca"

# Single combined monthly-resolution climate dataset spanning the full
# record (all cropped daily files aggregated to one file per month, one
# time dimension)
NOAA_MONTHLY_FILE = PROCESSED_DATA_DIR / "california_monthly_climate.nc"

# Final combined dataset: climate + PET + STI + SAPEI + SCDHI, all months
INDICES_FILE = PROCESSED_DATA_DIR / "california_monthly_indices.nc"

# Output directory for generated figures
FIGURES_DIR = ROOT / "figures"

# Review analyses (inference calibration, compound specificity, portfolio robustness)
REVIEW_DIR = FIGURES_DIR / "review"

# Raw EIA-923 (multi-sheet, all states)
RAW_DATA_ELECTRICITY_DIR = RAW_DATA_DIR / "electricity"
EIA_GENERATION_RAW_FILE = RAW_DATA_ELECTRICITY_DIR / "generation_monthly.xlsx"

# Processed California-only, source-grouped monthly generation table
GENERATION_MONTHLY_FILE = PROCESSED_DATA_DIR / "california_generation_monthly.csv"

# Raw CEC county-level electricity consumption workbook (restricted access,
# obtained by direct request -- not redistributable)
COUNTY_CONSUMPTION_XLSX = RAW_DATA_ELECTRICITY_DIR / "county_electricity_consumption_2008-2024.xlsx"

# Processed long-format monthly county consumption table (Date, CountyName, Sector, Consumption (MWh))
COUNTY_CONSUMPTION_MONTHLY_FILE = PROCESSED_DATA_DIR / "california_county_consumption_monthly.csv"

# Raw EIA-861M state-level monthly sales/revenue workbooks (two eras, different
# sheet names and sector-group sets)
EIA_DEMAND_RAW_FILE_1990_2009 = RAW_DATA_ELECTRICITY_DIR / "HS861M_1990-2009.xlsx"
EIA_DEMAND_RAW_FILE_2010_PRESENT = RAW_DATA_ELECTRICITY_DIR / "HS861M_2010-.xlsx"

# Processed statewide monthly total electricity sales (demand), combined across both eras
DEMAND_MONTHLY_FILE = PROCESSED_DATA_DIR / "california_demand_monthly.csv"

MANIFEST_PATH = DATA_DIR / "download_manifest.json"

_PIPELINE_DIRS = (
    RAW_DATA_DIR,
    INTERMEDIATE_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_CLIMATE_DIR,
    CA_BOUNDARY_DIR,
    CA_CLIMATE_REGIONS_DIR,
    CA_COUNTIES_DIR,
    NOAA_CA_DIR,
    FIGURES_DIR,
)


def ensure_dirs() -> None:
    """Create all pipeline directories if they don't already exist."""
    for d in _PIPELINE_DIRS:
        d.mkdir(parents=True, exist_ok=True)
