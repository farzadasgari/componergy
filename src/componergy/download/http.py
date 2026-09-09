"""Low-level HTTP download helper with checksum computation and resume-skip."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MB


def download_file(
        dataset_name: str,
        source_page: str,
        url: str,
        destination: Path,
        *,
        force: bool = False,
) -> tuple[Path, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        logger.info("Already downloaded, skipping: %s", dataset_name)
        return destination, _sha256(destination)

    logger.info("Downloading %s from %s (source: %s)", dataset_name, url, source_page)
    with requests.get(url, stream=True, timeout=60) as response:
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
