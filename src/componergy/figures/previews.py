"""Low-resolution JPEG previews of the 600 dpi figures the pipeline writes; sources are only ever read."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SOURCE_DPI = 600
PREVIEW_DPI = 100
PREVIEW_QUALITY = 90
PREVIEW_DIRNAME = "previews"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def preview_size(size: tuple, source_dpi: int = SOURCE_DPI, preview_dpi: int = PREVIEW_DPI) -> tuple:
    return tuple(max(1, round(n * preview_dpi / source_dpi)) for n in size)


def find_sources(figures_dir) -> list:
    root = Path(figures_dir)
    skip = root / PREVIEW_DIRNAME
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                  and skip not in p.parents)


def is_current(source: Path, preview: Path) -> bool:
    return preview.exists() and preview.stat().st_mtime >= source.stat().st_mtime


def make_preview(source, preview) -> tuple:
    """Write `preview` (JPEG, 100 dpi) from `source`; the file appears only once complete. Returns its pixel size."""
    source, preview = Path(source), Path(preview)
    preview.parent.mkdir(parents=True, exist_ok=True)
    tmp = preview.with_name(preview.name + ".tmp")
    with Image.open(source) as im:
        small = im.convert("RGB").resize(preview_size(im.size), Image.Resampling.LANCZOS)
    try:
        small.save(tmp, format="JPEG", quality=PREVIEW_QUALITY, dpi=(PREVIEW_DPI, PREVIEW_DPI))
        tmp.replace(preview)
    finally:
        tmp.unlink(missing_ok=True)
    return small.size


def build_previews(figures_dir, force: bool = False) -> dict:
    """Preview every image under `figures_dir` into `figures_dir/previews`, keeping sub-folders."""
    root = Path(figures_dir)
    out = {"written": [], "skipped": []}
    for src in find_sources(root):
        dst = (root / PREVIEW_DIRNAME / src.relative_to(root)).with_suffix(".jpg")
        if not force and is_current(src, dst):
            out["skipped"].append(dst)
            continue
        make_preview(src, dst)
        out["written"].append(dst)
    return out


def main(force: bool = False) -> None:
    from componergy.paths import FIGURES_DIR

    res = build_previews(FIGURES_DIR, force=force)
    print(f"previews written: {len(res['written'])}, up to date: {len(res['skipped'])} "
          f"(in {FIGURES_DIR / PREVIEW_DIRNAME})")
