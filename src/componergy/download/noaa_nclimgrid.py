"""
NOAA nClimGrid-Daily fetcher.

Builds one DatasetTask per (year, month) in the requested range.
Files are CONUS-wide gridded NetCDF (~one file per calendar month);
see registry.download_all() for how these are run alongside the other
registered datasets in a single pipeline run.

Months at or after the current calendar month are skipped -- NOAA has
not published them yet by definition. Any remaining publication lag
beyond that (recent-but-past months not yet released) is handled at
download time in http.py, which distinguishes "not yet available" (404)
from a real failure.
"""

from __future__ import annotations

from datetime import date

from componergy.download.tasks import DatasetTask
from componergy.paths import RAW_DATA_CLIMATE_DIR

BASE_URL = "https://www.ncei.noaa.gov/data/nclimgrid-daily/access/grids"
SOURCE_PAGE = "https://www.ncei.noaa.gov/products/land-based-station/nclimgrid-daily"


def build_tasks(start_year: int, end_year: int) -> list[DatasetTask]:
    """
    Build one DatasetTask per calendar month in [start_year, end_year]
    (inclusive), excluding months at or after the current calendar month.
    """
    today = date.today()
    tasks = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if (year, month) >= (today.year, today.month):
                continue
            fname = f"ncdd-{year}{month:02d}-grd-scaled.nc"
            tasks.append(
                DatasetTask(
                    name=f"NOAA nClimGrid-Daily {year}-{month:02d}",
                    source_page=SOURCE_PAGE,
                    url=f"{BASE_URL}/{year}/{fname}",
                    destination=RAW_DATA_CLIMATE_DIR / fname,
                )
            )
    return tasks
