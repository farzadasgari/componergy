"""The end-to-end workflow as data: stages, what each needs, and how to run it."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Stage:
    name: str
    script: str
    description: str
    needs: tuple = ()
    accepts_baseline: bool = False
    accepts_demand_source: bool = False
    extra: tuple = field(default_factory=tuple)


STAGES = (
    Stage("download", "run_download_pipeline", "NOAA nClimGrid-Daily 1951-2026 + boundaries (network)"),
    Stage("crop", "run_crop_pipeline", "clip daily grids to California"),
    Stage("verify", "run_verify_crop", "detect truncated cropped files"),
    Stage("aggregate", "run_aggregate_pipeline", "daily -> monthly climate"),
    Stage("indices", "run_indices_pipeline", "PET, STI, SAPEI, SCDHI (slow)"),
    Stage("generation", "run_generation_pipeline", "EIA-923 monthly generation by source",
          needs=("EIA_GENERATION_RAW_FILE",)),
    Stage("demand", "run_demand_pipeline", "EIA-861M monthly retail sales (public cross-check demand)",
          needs=("EIA_DEMAND_RAW_FILE_1990_2009", "EIA_DEMAND_RAW_FILE_2010_PRESENT")),
    Stage("consumption", "run_consumption_pipeline", "CEC county consumption (RESTRICTED)",
          needs=("COUNTY_CONSUMPTION_XLSX",)),
    Stage("event-check", "run_event_definition_check", "event-count sensitivity to baseline and smoothing (90/10/10)",
          needs=("INDICES_FILE",)),
    Stage("sensitivity", "run_sensitivity_checks", "index construction + county-to-region sensitivity",
          needs=("INDICES_FILE",)),
    Stage("figure-01", "run_figure_01", "trend maps", needs=("INDICES_FILE",)),
    Stage("figure-02", "run_figure_02", "generation mix", needs=("INDICES_FILE", "GENERATION_MONTHLY_FILE"),
          accepts_baseline=True),
    Stage("figure-03", "run_figure_03", "regional lag scan + relative importance",
          needs=("INDICES_FILE", "COUNTY_CONSUMPTION_XLSX")),
    Stage("figure-04", "run_figure_04", "zonal consumption", needs=("INDICES_FILE", "COUNTY_CONSUMPTION_XLSX"),
          accepts_baseline=True),
    Stage("figure-05", "run_figure_05", "sector fingerprints", needs=("INDICES_FILE", "COUNTY_CONSUMPTION_XLSX"),
          accepts_baseline=True),
    Stage("figure-06", "run_figure_06", "imbalance", needs=("INDICES_FILE", "GENERATION_MONTHLY_FILE"),
          accepts_baseline=True, accepts_demand_source=True),
    Stage("figure-07", "run_figure_07", "systemic vulnerability", needs=("INDICES_FILE", "COUNTY_CONSUMPTION_XLSX"),
          accepts_baseline=True),
    Stage("supplementary", "run_supplementary_figures", "S1-S7", needs=("INDICES_FILE",)),
    Stage("review", "run_review_analyses", "review-response analyses + study-period table", needs=("INDICES_FILE",)),
    Stage("manifest", "make_run_manifest", "checksums, versions, seeds"),
)
STAGE_NAMES = tuple(s.name for s in STAGES)


def plan(only=None, start=None, end=None, skip_missing=False, baseline=None, demand_source=None,
         exists=None) -> list:
    """Ordered list of {stage, command, status}; `exists(name)` says whether a paths constant is on disk."""
    if exists is None:
        from componergy import paths
        exists = lambda n: getattr(paths, n).exists()  # noqa: E731
    names = list(STAGE_NAMES)
    for label, val in (("only", only), ("start", start), ("end", end)):
        for v in ([] if val is None else (val if isinstance(val, (list, tuple)) else [val])):
            if v not in names:
                raise ValueError(f"unknown stage {v!r} for --{label}; choose from {names}")
    lo = names.index(start) if start else 0
    hi = names.index(end) if end else len(names) - 1
    selected = [s for s in STAGES[lo:hi + 1] if (not only or s.name in only)]

    out = []
    for s in selected:
        cmd = [sys.executable, "-m", f"scripts.{s.script}"]
        if baseline and s.accepts_baseline:
            cmd += ["--baseline", baseline]
        if demand_source and s.accepts_demand_source:
            cmd += ["--demand-source", demand_source]
        missing = [n for n in s.needs if not exists(n)]
        status = "run" if not missing else f"skip (missing: {', '.join(missing)})"
        if missing and not skip_missing:
            status = f"BLOCKED (missing: {', '.join(missing)})"
        out.append({"stage": s.name, "command": cmd, "status": status, "description": s.description})
    return out
