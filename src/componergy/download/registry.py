"""Central orchestration for the componergy data-acquisition pipeline.

Runs all registered download tasks -- currently the California state
boundary (single file) and NOAA nClimGrid-Daily (many files, downloaded
in parallel) -- and records every downloaded file in a JSON manifest.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from componergy.download import boundaries, noaa_nclimgrid
from componergy.download.http import download_file
from componergy.download.manifest import append_manifest
from componergy.download.tasks import DatasetTask
from componergy.paths import ensure_dirs

logger = logging.getLogger(__name__)


def _run_task(task: DatasetTask) -> None:
    _, digest = download_file(
        dataset_name=task.name,
        source_page=task.source_page,
        url=task.url,
        destination=task.destination,
    )
    append_manifest(name=task.name, url=task.url, destination=task.destination, digest=digest)


def download_all(
        max_workers: int = 4,
        include_noaa: bool = True,
        noaa_start_year: int = 1991,
        noaa_end_year: int = 2025,
) -> None:
    """Run the full download pipeline."""
    ensure_dirs()

    logger.info("Fetching California boundary")
    boundaries.get_california_boundary()

    if not include_noaa:
        logger.info("Skipping NOAA nClimGrid downloads (--skip-noaa)")
        return

    tasks = noaa_nclimgrid.build_tasks(noaa_start_year, noaa_end_year)
    logger.info("Downloading %d NOAA nClimGrid-Daily files", len(tasks))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except Exception:
                logger.exception("Failed to download %s", task.name)
