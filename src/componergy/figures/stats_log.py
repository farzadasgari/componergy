from __future__ import annotations

import json
from pathlib import Path


def write_figure_log(path, sections: dict) -> None:
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
