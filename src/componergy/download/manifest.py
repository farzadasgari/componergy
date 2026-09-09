from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from componergy.paths import MANIFEST_PATH, ensure_dirs


def append_manifest(name: str, url: str, destination: Path, digest: str) -> None:
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