"""Run manifest: what exactly produced a set of results."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = ("numpy", "pandas", "xarray", "scipy", "matplotlib", "geopandas", "regionmask", "netCDF4",
            "rioxarray", "dask", "statsmodels", "joblib", "cartopy", "shapely", "pyproj")
SEEDS = {"bootstrap_raf": 42, "superadditivity": 42, "lag_scan_randomization": 42,
         "relative_importance_bootstrap": 42, "return_period_bootstrap": 42, "copula_selection": 0}
HASH_LIMIT_BYTES = 500 * 1024 * 1024


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def directory_fingerprint(path, pattern: str = "*") -> dict:
    files = sorted(p for p in Path(path).rglob(pattern) if p.is_file())
    listing = "\n".join(f"{p.relative_to(path)}:{p.stat().st_size}" for p in files)
    return {"n_files": len(files), "total_bytes": int(sum(p.stat().st_size for p in files)),
            "listing_sha256": hashlib.sha256(listing.encode()).hexdigest()}


def describe_path(path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    if p.is_dir():
        return {"path": str(p), "exists": True, "kind": "directory", **directory_fingerprint(p)}
    st = p.stat()
    out = {"path": str(p), "exists": True, "kind": "file", "size_bytes": int(st.st_size),
           "modified_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")}
    out["sha256"] = sha256_file(p) if st.st_size <= HASH_LIMIT_BYTES else "skipped (file larger than limit)"
    return out


def package_versions(names=PACKAGES) -> dict:
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def git_info(root) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20,
                                  check=True).stdout.strip()
        except Exception:
            return None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "describe": run("describe", "--tags", "--always"),
            "dirty": (bool(status) if status is not None else None)}


def build_manifest(inputs: dict, outputs: dict, root, seeds: dict | None = None, extra: dict | None = None) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": git_info(root),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": package_versions(),
        "seeds": SEEDS if seeds is None else seeds,
        "inputs": {k: describe_path(v) for k, v in inputs.items()},
        "outputs": {k: describe_path(v) for k, v in outputs.items()},
        **(extra or {}),
    }


def compare_manifests(a: dict, b: dict) -> dict:
    """What changed between two manifests: inputs/outputs whose checksum or fingerprint differs, and package versions."""

    def key(d):
        return d.get("sha256") or d.get("listing_sha256") or d.get("exists")

    changed = {}
    for section in ("inputs", "outputs"):
        for name in sorted(set(a.get(section, {})) | set(b.get(section, {}))):
            ka, kb = key(a.get(section, {}).get(name, {})), key(b.get(section, {}).get(name, {}))
            if ka != kb:
                changed[f"{section}.{name}"] = {"a": ka, "b": kb}
    pk = {n: (a["packages"].get(n), b["packages"].get(n)) for n in set(a["packages"]) | set(b["packages"])
          if a["packages"].get(n) != b["packages"].get(n)}
    return {"changed_files": changed, "changed_packages": pk,
            "same_commit": a["code"].get("commit") == b["code"].get("commit")}
