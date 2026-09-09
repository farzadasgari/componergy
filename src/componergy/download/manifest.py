"""Provenance log of every file downloaded by the pipeline.

Kept as its own module (rather than living in registry.py) so that both
registry.py and any individual dataset module (e.g. boundaries.py) can
record downloads without those modules having to import each other.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from componergy.paths import MANIFEST_PATH, ensure_dirs


def append_manifest(name: str, url: str, destination: Path, digest: str) -> None:
    """Append a record of one downloaded file to the JSON manifest."""
    ensure_dirs()

    entry = {
        "name": name,
        "url": url,
        "destination": str(destination),
        "sha256": digest,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }

    manifest = []
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())

    manifest.append(entry)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
