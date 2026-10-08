# src/componergy/analysis/index_diagnostics.py
"""Diagnostics for how the climate indices are constructed"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def standardized_precip_index(prcp: pd.Series, scale: int = 3) -> pd.Series:
    """Nonparametric SPI: `scale`-month precipitation totals standardized within each calendar month."""
    acc = prcp.astype(float).rolling(scale, min_periods=scale).sum()
    out = pd.Series(np.nan, index=acc.index)
    for m in range(1, 13):
        sel = acc[(acc.index.month == m) & acc.notna()]
        if len(sel) < 5:
            continue
        ranks = sel.rank(method="average")
        p = (ranks - 0.44) / (len(sel) + 0.12)
        out.loc[sel.index] = stats.norm.ppf(p.values)
    return out.rename(f"spi_{scale}m")


def temperature_dependence(sti: pd.Series, sapei: pd.Series, spi: pd.Series) -> dict:
    """How much of each drought index moves with STI, over the months all three share."""
    df = pd.concat([sti.rename("sti"), sapei.rename("sapei"), spi.rename("spi")], axis=1).dropna()
    if len(df) < 24:
        return {"n_months": int(len(df)), "verdict": "too few overlapping months"}
    r_sapei = float(df["sti"].corr(df["sapei"]))
    r_spi = float(df["sti"].corr(df["spi"]))
    excess = r_sapei ** 2 - r_spi ** 2
    return {"n_months": int(len(df)), "corr_STI_SAPEI": r_sapei, "corr_STI_SPI": r_spi,
            "r2_SAPEI_on_STI": r_sapei ** 2, "r2_SPI_on_STI": r_spi ** 2,
            "excess_r2_from_temperature": excess, "corr_SAPEI_SPI": float(df["sapei"].corr(df["spi"])),
            "verdict": ("SAPEI tracks STI materially more than a precipitation-only index does: part of the "
                        "drought signal is evaporative demand, so temperature reaches SCDHI twice"
                        if excess > 0.05 else
                        "SAPEI tracks STI about as much as a precipitation-only index does: little evidence "
                        "that temperature enters the compound index twice")}


def drought_definition_agreement(sapei: pd.Series, spi: pd.Series, q: float = 0.10, window=None,
                                 smooth_months: int = 3) -> dict:
    """Drought months (lowest `q` quantile, full-record baseline, same smoothing as the figures): SAPEI vs SPI."""

    def flags(s):
        sm = s.astype(float).rolling(smooth_months, center=True, min_periods=1).mean()
        thr = float(np.nanquantile(sm.values, q))
        f = sm < thr
        return f if window is None else f.reindex(window).fillna(False)

    a, b = flags(sapei), flags(spi)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    inter, union = int((a & b).sum()), int((a | b).sum())
    return {"n_months": int(len(common)), "drought_months_SAPEI": int(a.sum()), "drought_months_SPI": int(b.sum()),
            "both": inter, "SAPEI_only": int((a & ~b).sum()), "SPI_only": int((b & ~a).sum()),
            "jaccard": float(inter / union) if union else np.nan}
