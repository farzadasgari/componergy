import numpy as np
import pandas as pd

from componergy.analysis.risk_sensitivity import (
    build_risk_variants,
    compare_risk_variants,
    double_calm_share,
    REFERENCE,
)


def _idx(n=120):
    return pd.date_range("2010-01-01", periods=n, freq="MS")
