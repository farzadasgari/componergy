from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
PROCESSED_DIR = DATA_DIR / "processed"

SCRIPTS_DIR = ROOT / "scripts"

FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"
OUTPUTS_DIR = ROOT / "outputs"
LOGS_DIR = ROOT / "logs"

NOAA_RAW_DIR = RAW_DIR / "noaa-nclimgrid"
NOAA_CA_DIR = INTERMEDIATE_DIR / "noaa-nclimgrid-ca"
NOAA_MONTHLY_FILE = PROCESSED_DIR / "monthly_ca_1951_2025.nc"

CENSUS_DIR = RAW_DIR / "census-shape"
STATE_SHAPEFILE = CENSUS_DIR / "tl_2025_us_state.shp"