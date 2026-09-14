"""Structured, dual-format (human-readable + machine-readable) figure summary logging."""
from __future__ import annotations

import json
from pathlib import Path


def write_figure_log(path, sections: dict) -> None:
    """
    Write `sections` as both a plain-text .log and a .json file at `path`.

    Both files are derived from the same dict, so the two never drift out
    of sync with each other. Each top-level key is a section title;
    values that are themselves dicts are written as key: value lines,
    everything else is written as-is.

    Parameters
    ----------
    path : path-like
        Output path; the extension is replaced with .log and .json.
    sections : dict
        {section_title: content}, where content is a dict or any
        JSON-serializable value.
    """
    path = Path(path)
    txt_path = path.with_suffix(".log")
    json_path = path.with_suffix(".json")

    with open(txt_path, "w", encoding="utf-8") as f:
        for title, content in sections.items():
            f.write(f"---- {title} ----\n")
            if isinstance(content, dict):
                for key, value in content.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write(f"{content}\n")
            f.write("\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, default=str)
