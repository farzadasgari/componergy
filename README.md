<div align="center">

# Compowater 🌊🌡️
## Compound Drought–Heat Extremes and Urban Water Resilience in California

<a href="https://github.com/farzadasgari/compowater/blob/main/LICENSE"> <img src="https://img.shields.io/badge/License-Apache%202.0-2ea44f?style=for-the-badge&logo=apache&logoColor=white" /> </a>
<img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Status-Research%20Development-ff9800?style=for-the-badge&logo=academia&logoColor=white" />
<a href="https://github.com/farzadasgari/compowater/actions/workflows/ci.yml"> <img src="https://img.shields.io/github/actions/workflow/status/farzadasgari/compowater/ci.yml?style=for-the-badge&logo=githubactions&logoColor=white" /> </a>
<img src="https://img.shields.io/badge/code%20style-black-000000?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/lint-ruff-000000?style=for-the-badge&logo=ruff&logoColor=white" />

</div>

---

## Table of Contents
- [Overview](#overview)
- [Project Status](#project-status)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Code Quality: Linting & Formatting](#code-quality-linting--formatting)
- [Testing](#testing)
- [Reproducibility & Provenance](#reproducibility--provenance)
- [Contributing](#contributing)
- [How to Cite](#how-to-cite)
- [License](#license)
- [Contact](#contact)
- [Research Team](#research-team)

---

## Overview

Compowater investigates how **compound extreme events** — concurrent
drought and heatwave conditions — affect urban water supply and demand
across California. Individually, drought and heat stress are well
studied; their *joint* occurrence can amplify strain on urban water
systems in ways that univariate hazard analyses miss, with direct
implications for utility planning and state-level water policy. This
repository contains the full, open-source, reproducible data and
analysis pipeline supporting that work — from raw data acquisition
through to the figures and results in the associated manuscript.

## Project Status

🚧 Under active development. Current progress:

## Repository Structure

```
compowater/
├── .github/
│   └── workflows/
│       └── ci.yml               # test, lint, and format checks on every push/PR
├── CHANGELOG.md
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── config/
│   └── datasets.yaml             # Declarative registry of static-URL datasets
├── data/                          # Not tracked by git — see .gitignore
│   ├── intermediate/
│   ├── processed/
│   └── raw/
│       ├── _manifest.jsonl       # Provenance log: source, hash, download timestamp
│       ├── reservoirs/           # CA DWR reservoir data (CSV)
│       └── urban_water/          # CA urban water supply/demand (CSV)
│                                  # (climate/ NOAA NetCDFs also land under raw/)
├── environment.yml                # Conda/conda-forge environment spec
├── notebooks/
│   └── README.md                  # .ipynb files untracked pre-publication
├── pyproject.toml                 # Package metadata, ruff/black config
├── requirements.txt                # Top-level pip dependencies
├── requirements-lock.txt           # Fully pinned, exact-version pip environment
├── scripts/
│   └── run_download_pipeline.py   # CLI entry point for data acquisition
├── src/
│   └── compowater/
│       ├── __init__.py
│       ├── paths.py                # Centralized path definitions
│       └── download/               # Retry-enabled, checksum-verified fetchers
│           ├── exceptions.py       # ResourceNotFoundError, etc.
│           ├── http.py             # Core download_file with atomic writes
│           ├── noaa_nclimgrid.py   # NOAA task-builder
│           ├── parallel.py         # Concurrent download_many
│           ├── registry.py         # YAML + NOAA task merging, manifest logging
│           └── tasks.py            # Shared DatasetTask definition
└── tests/
    ├── test_paths.py
    └── download/
        ├── test_http.py
        ├── test_noaa_nclimgrid.py
        ├── test_parallel.py
        ├── test_registry.py
        └── test_tasks.py
```

## Environment Setup

Choose **one** path — both install the same dependencies.

### Option A — pip / venv

```bash
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash
pip install -r requirements-lock.txt
pip install -e .
```

### Option B — conda / Miniforge

Recommended if you plan to work with the geospatial stack
(`geopandas`, `rasterio`, `rioxarray`, `cartopy`), since these wrap
compiled C libraries (GDAL/GEOS/PROJ) that conda-forge distributes as
pre-built binaries.

1. Install [Miniforge](https://github.com/conda-forge/miniforge).
2. From the repo root:
   ```bash
   conda env create -f environment.yml
   conda activate compowater
   pip install -e .
   ```

Either path: `pip install -e .` installs `compowater` in editable
mode, so `import compowater` works from anywhere and local code edits
take effect immediately.

**Adding a new dependency?** Add it to `requirements.txt` *and*
`environment.yml`, then regenerate the lock file:
```bash
pip freeze | grep -v compowater > requirements-lock.txt
```

## Code Quality: Linting & Formatting

This project uses [ruff](https://docs.astral.sh/ruff/) for linting
(unused imports, undefined names, import ordering) and
[black](https://black.readthedocs.io/) for automatic code formatting.
Both run in CI on every push and pull request; to run them locally:

```bash
pip install ruff black

ruff check src/ scripts/ tests/          # check for lint issues
ruff check --fix src/ scripts/ tests/    # auto-fix what's fixable

black --check src/ scripts/ tests/       # check formatting only
black src/ scripts/ tests/               # auto-format in place
```

Configuration for both tools lives in `pyproject.toml` under
`[tool.ruff]` and `[tool.black]`.

## Testing

The full download/data-acquisition layer has unit test coverage under
`tests/`, using `pytest` and `requests-mock` — no test touches the
real network or the real project data folders.

```bash
pip install pytest requests-mock
pytest -v
```

`.github/workflows/ci.yml` runs three independent checks on every
push and pull request:
- **test** — the full pytest suite
- **lint** — ruff
- **format** — black --check

## Reproducibility & Provenance

Every file the pipeline downloads is recorded in
`data/raw/_manifest.jsonl`: source URL, destination path, SHA-256
hash, and UTC download timestamp. This matters because **two of our
three sources are "living" datasets** — the CA DWR and CA SWRCB CSVs
are overwritten in place on a monthly cadence. A fixed checksum can't
verify these against a known-good value the way it can for a static
archive; instead, the manifest is what lets us pin the *exact
snapshot* used for a given analysis. Data versioning via DVC is
planned to formalize this snapshotting further (see roadmap above).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report a
reproducibility issue or suggest a fix, and
[CHANGELOG.md](CHANGELOG.md) for a record of notable changes.

## How to Cite

This repository accompanies an in-preparation manuscript. Full
citation details (DOI, BibTeX) will be added upon submission or
publication. A machine-readable citation is also available via
[CITATION.cff](CITATION.cff) (see GitHub's "Cite this repository"
button). In the meantime, please cite the repository directly:

> Asgari, F., Mohajeri, S.H., & Nikoo, M.R. (2026). *Compowater:
> Compound Drought–Heat Extremes and Urban Water Resilience*
> [Software repository]. https://github.com/farzadasgari/compowater

## License

This project's code is licensed under the Apache License 2.0. See
[LICENSE](LICENSE) for details.

## Contact

For any inquiries, please contact:
- std_farzad.asgari@khu.ac.ir
- khufarzadasgari@gmail.com

## Research Team

### Farzad Asgari
<div align="center">

[![github](https://img.shields.io/badge/GitHub-6e5494?style=for-the-badge&logo=github&logoColor=white)](https://github.com/farzadasgari)
[![Google Scholar Badge](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=fff&style=for-the-badge)](https://scholar.google.com/citations?user=Rhue_kkAAAAJ&hl=en)
[![linkedin](https://custom-icon-badges.demolab.com/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin-white&logoColor=white)](https://www.linkedin.com/in/farzad-asgari)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--3800--0408-A6CE39?logo=orcid&logoColor=fff&style=for-the-badge)](https://orcid.org/0009-0008-3800-0408)

</div>

### Seyed Hossein Mohajeri
<div align="center">

[![Google Scholar Badge](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=fff&style=for-the-badge)](https://scholar.google.com/citations?user=E8PFUBEAAAAJ&hl=en)
[![linkedin](https://custom-icon-badges.demolab.com/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin-white&logoColor=white)](https://www.linkedin.com/in/hossein-mohajeri)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--8056--293X-A6CE39?logo=orcid&logoColor=fff&style=for-the-badge)](https://orcid.org/0000-0002-8056-293X)

</div>

### Mohammad Reza Nikoo
<div align="center">

[![Google Scholar Badge](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=fff&style=for-the-badge)](https://scholar.google.com/citations?hl=en&user=tbsw9yAAAAAJ)
[![linkedin](https://custom-icon-badges.demolab.com/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin-white&logoColor=white)](https://www.linkedin.com/in/mohammad-reza-nikoo-2a819b55)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--3740--4389-A6CE39?logo=orcid&logoColor=fff&style=for-the-badge)](https://orcid.org/0000-0002-3740-4389)

</div>
