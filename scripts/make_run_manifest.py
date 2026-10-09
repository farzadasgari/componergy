"""Write figures/run_manifest.json: code version, environment, seeds, and checksums of every input and output."""

from __future__ import annotations

import json


def main() -> None:
    from componergy import paths
    from componergy.provenance import build_manifest

    def p(name):
        return getattr(paths, name, None)

    inputs = {n: p(n) for n in ("RAW_DATA_CLIMATE_DIR", "EIA_GENERATION_RAW_FILE", "EIA_DEMAND_RAW_FILE_1990_2009",
                                "EIA_DEMAND_RAW_FILE_2010_PRESENT", "COUNTY_CONSUMPTION_XLSX")}
    outputs = {n: p(n) for n in ("NOAA_MONTHLY_FILE", "INDICES_FILE", "GENERATION_MONTHLY_FILE",
                                 "DEMAND_MONTHLY_FILE", "COUNTY_CONSUMPTION_MONTHLY_FILE", "FIGURES_DIR")}
    manifest = build_manifest({k: v for k, v in inputs.items() if v is not None},
                              {k: v for k, v in outputs.items() if v is not None}, root=paths.ROOT)
    out = paths.FIGURES_DIR / "run_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"wrote {out}")
    print("commit:", manifest["code"]["commit"], "| dirty:", manifest["code"]["dirty"])


if __name__ == "__main__":
    main()
