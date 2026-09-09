"""
Central orchestration for the componergy data-acquisition pipeline.

Runs all registered download tasks -- currently the California state
boundary (single file) and NOAA nClimGrid-Daily (many files, downloaded
in parallel) -- and records every downloaded file in a JSON manifest.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from componergy.download import boundaries, noaa_nclimgrid
from componergy.download.http import DatasetNotAvailableError, download_file
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
        noaa_start_year: int = 1951,
        noaa_end_year: int = 2026,
) -> None:
    """Run the full download pipeline."""
    print(f"starting componergy download pipeline (max_workers={max_workers})")
    ensure_dirs()

    print("fetching California boundary...")
    boundaries.get_california_boundary()
    print("California boundary ready.")

    if not include_noaa:
        print("skipping NOAA nClimGrid downloads (--skip-noaa flag set).")
        return

    tasks = noaa_nclimgrid.build_tasks(noaa_start_year, noaa_end_year)
    print(f"queued {len(tasks)} NOAA nClimGrid-Daily files ({noaa_start_year}-{noaa_end_year}).")

    succeeded = 0
    unavailable: list[str] = []
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks}
        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="downloading NOAA nClimGrid-Daily",
            unit="file",
        )
        for future in progress:
            task = futures[future]
            try:
                future.result()
                succeeded += 1
            except DatasetNotAvailableError:
                unavailable.append(task.name)
                progress.write(f"not yet available: {task.name}")
            except Exception:
                failed.append(task.name)
                logger.exception("failed to download %s", task.name)
                progress.write(f"FAILED: {task.name}")

    print(
        f"download pipeline complete: {succeeded} succeeded, "
        f"{len(unavailable)} not yet available, {len(failed)} failed."
    )
    if unavailable:
        print("not yet available (re-run later once the source publishes them):")
        for name in unavailable:
            print(f"  - {name}")
    if failed:
        print("failed downloads:")
        for name in failed:
            print(f"  - {name}")
