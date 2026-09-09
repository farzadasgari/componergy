"""Low-level HTTP download helper with checksum computation and resume-skip."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MB


class DatasetNotAvailableError(Exception):
    """
    Raised when a dataset URL returns HTTP 404.

    Distinct from other failures: this generally means the source hasn't
    published the file yet (e.g. a very recent month still in QC), not
    that something is broken. Callers can treat this as "try again later"
    rather than a hard failure.
    """


def download_file(
        dataset_name: str,
        source_page: str,
        url: str,
        destination: Path,
        *,
        force: bool = False,
) -> tuple[Path, str]:
    """
    Download a single file and return (path, sha256 digest).

    Skips the actual download (but still computes the digest) if
    `destination` already exists and `force` is False, so pipeline
    re-runs are cheap and idempotent.

    Raises
    ------
    DatasetNotAvailableError
        If the URL returns HTTP 404 (not yet published).
    requests.exceptions.HTTPError
        For any other non-2xx response.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        logger.info("already downloaded, skipping: %s", dataset_name)
        return destination, _sha256(destination)

    logger.info("downloading %s from %s (source: %s)", dataset_name, url, source_page)
    with requests.get(url, stream=True, timeout=60) as response:
        if response.status_code == 404:
            raise DatasetNotAvailableError(
                f"{dataset_name}: {url} returned 404 (not yet published by the source)"
            )
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    return destination, _sha256(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
