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
    Stage("event-check", "run_event_definition_check", "which event definition reproduces the quoted counts",
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
