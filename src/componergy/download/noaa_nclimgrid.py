from __future__ import annotations

from componergy.download.tasks import DatasetTask
from componergy.paths import RAW_DATA_CLIMATE_DIR

BASE_URL = "https://www.ncei.noaa.gov/data/nclimgrid-daily/access/grids"
SOURCE_PAGE = "https://www.ncei.noaa.gov/products/land-based-station/nclimgrid-daily"


def build_tasks(start_year: int, end_year: int) -> list[DatasetTask]:
    tasks = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
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