# Data Directory

This directory contains the datasets required to reproduce the analyses presented in the **Componergy** framework.

## Directory structure

```text
data/
├── raw/
├── intermediate/
└── processed/
```

### `raw/`

Contains the original datasets obtained directly from external providers without modification. Examples include NOAA nClimGrid climate data, U.S. Census shapefiles, California Energy Commission (CEC) electricity demand data, and U.S. Energy Information Administration (EIA) generation data.

### `intermediate/`

Contains temporary datasets generated during preprocessing, such as cropped spatial subsets, reformatted files, and intermediate products used in subsequent analyses.

These files can always be regenerated from the raw datasets using the preprocessing scripts.

### `processed/`

Contains finalized datasets produced by the preprocessing workflow that are used directly for statistical analyses, visualization, and manuscript figures.

## Data availability

Raw datasets are **not distributed** with this repository because of their large size and because they are provided by third-party data providers.

All required datasets are publicly available from their respective sources.

The download scripts located in `scripts/download/` automatically retrieve the publicly available datasets required to reproduce the study.

## Reproducibility

The complete preprocessing workflow is documented in the repository and can be reproduced using the provided download and preprocessing scripts.
