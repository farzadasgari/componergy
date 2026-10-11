import numpy as np
import pandas as pd
import pytest

from componergy.analysis.relative_importance import (
    lmg_decomposition,
    variance_inflation,
    lmg_table,
    incremental_r2_block_permutation,
)


def _data(n=204, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    return a, b, rng
