# Exploratory / superseded analysis

The scripts in this folder were part of the working research process but are
**not** part of the pipeline that produced the figures and results in the
manuscript. They are kept here for transparency and provenance only.

None of the final published figures (`figures/figure_01_trend.py` through
`figures/figure_07_risk_surface.py`) read any file produced by these scripts —
this was verified by cross-referencing every input path in the final figure
scripts against every output path in this folder.

**To reproduce the manuscript's results, you do not need anything in this
folder.** Use the pipeline described in the main `README.md` instead
(`scripts/download` → `scripts/preprocessing` → `src/componergy` →
`figures/`).

## What's here and why it exists

| File | What it was for |
|---|---|
| `ultimate.py` | Original merge script. Its "monthly" half was adapted into `scripts/preprocessing/merge_california_monthly.py` (the version actually used). Its other half built `california_warmest.nc`, which no final figure reads. |
| `sdhiwarmest.py`, `sceiwarmest.py` | Earlier "warmest 3-month window" variants of the drought/heat indices, feeding the unused half of `ultimate.py` above. |
| `cecdataset.py` | An earlier cleaning pass over the CEC county-consumption file. The final figures read that raw file directly instead. |
| `countyelectricity.py`, `electriciticaiso.py`, `emissionepa.py`, `californiadataset.py` | Built a combined dataset (CAISO load, EPA emissions, county electricity) for exploratory impact analysis — not used in the final figures. |
| `impacts.py`, `impacts2.py`, `generationimpact.py`, `comparsion_between_indices.py`, `regionalimpacts.py`, `regionalimpacts2.py` | Exploratory statistical/impact analyses on the combined dataset above. |
| `hovmoller.py`, `regionalcalifornia.py`, `regionalmaps.py`, `spatialelectricity.py`, `spatialemission.py`, `visualizations.py` | Earlier exploratory plotting scripts, superseded by the final `figures/` scripts. |

If you want to keep exploring any of this later, note that these scripts still
have their original hardcoded relative paths (`../dataset/...`) and have not
been fixed, tested, or wired into `paths.py`.
