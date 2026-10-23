# src/componergy/review/freeze.py
"""SHA-256 freeze check: prove that a review run left the published outputs untouched."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from componergy.provenance import sha256_file


def hash_tree(root, patterns=("figures/figure_0*", "figures/robustness", "data/processed")) -> dict:
    """{relative path: sha256} of every file under the pattern directories (patterns are relative to `root`)."""
    root = Path(root)
    out = {}
    for pat in patterns:
        for base in sorted(root.glob(pat)):
            files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
            for f in files:
                out[f.relative_to(root).as_posix()] = sha256_file(f)
    return dict(sorted(out.items()))


def digest(hashes: dict) -> str:
    """One digest of a {path: hash} map; independent of insertion order."""
    h = hashlib.sha256()
    for k in sorted(hashes):
        h.update(f"{k}\0{hashes[k]}\n".encode())
    return h.hexdigest()


def diff_hashes(before: dict, after: dict) -> dict:
    changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    return {"changed": changed, "added": sorted(after.keys() - before.keys()),
            "removed": sorted(before.keys() - after.keys())}


def write_freeze(path, hashes: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"digest": digest(hashes), "n_files": len(hashes), "files": hashes}, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)
